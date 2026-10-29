"""Record real G1 arm command and state DDS packets without publishing commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


JOINTS = {
    12: "waist_yaw",
    15: "left_shoulder_pitch",
    16: "left_shoulder_roll",
    17: "left_shoulder_yaw",
    18: "left_elbow",
    19: "left_wrist_roll",
    20: "left_wrist_pitch",
    21: "left_wrist_yaw",
    22: "right_shoulder_pitch",
    23: "right_shoulder_roll",
    24: "right_shoulder_yaw",
    25: "right_elbow",
    26: "right_wrist_roll",
    27: "right_wrist_pitch",
    28: "right_wrist_yaw",
}
ARM_SLOTS = tuple(range(15, 29))


def _command_event(msg: LowCmd_, topic: str) -> dict:
    return {
        "time_unix_ns": time.time_ns(),
        "time_monotonic_ns": time.monotonic_ns(),
        "stream": "command",
        "topic": topic,
        "crc_valid": CRC().Crc(msg) == msg.crc,
        "mode_pr": int(msg.mode_pr),
        "mode_machine": int(msg.mode_machine),
        "arm_sdk_weight": float(msg.motor_cmd[29].q) if topic == "rt/arm_sdk" else None,
        "joints": {
            JOINTS[slot]: {
                "slot": slot,
                "mode": int(msg.motor_cmd[slot].mode),
                "q": float(msg.motor_cmd[slot].q),
                "dq": float(msg.motor_cmd[slot].dq),
                "kp": float(msg.motor_cmd[slot].kp),
                "kd": float(msg.motor_cmd[slot].kd),
                "tau": float(msg.motor_cmd[slot].tau),
            }
            for slot in JOINTS
        },
    }


def _state_event(msg: LowState_) -> dict:
    return {
        "time_unix_ns": time.time_ns(),
        "time_monotonic_ns": time.monotonic_ns(),
        "stream": "state",
        "topic": "rt/lowstate",
        "crc_valid": CRC().Crc(msg) == msg.crc,
        "mode_pr": int(msg.mode_pr),
        "mode_machine": int(msg.mode_machine),
        "tick": int(msg.tick),
        "joints": {
            JOINTS[slot]: {
                "slot": slot,
                "mode": int(msg.motor_state[slot].mode),
                "q": float(msg.motor_state[slot].q),
                "dq": float(msg.motor_state[slot].dq),
                "tau_est": float(msg.motor_state[slot].tau_est),
                "temperature": [int(value) for value in msg.motor_state[slot].temperature],
                "motorstate": int(msg.motor_state[slot].motorstate),
            }
            for slot in JOINTS
        },
    }


def _maximum(event: dict | None, field: str, slots=ARM_SLOTS) -> float | None:
    if event is None:
        return None
    return max(abs(event["joints"][JOINTS[slot]][field]) for slot in slots)


def _position_error(command: dict | None, state: dict | None) -> float | None:
    if command is None or state is None:
        return None
    return max(
        abs(command["joints"][JOINTS[slot]]["q"] - state["joints"][JOINTS[slot]]["q"])
        for slot in ARM_SLOTS
    )


def _format(value: float | None) -> str:
    return "waiting" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", default="", help="Interface connected to the robot")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain; the real robot normally uses 0")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record; 0 records until Ctrl+C")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path; defaults to logs/g1_arm_dds_<timestamp>.jsonl",
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must be nonnegative")

    output = args.output or Path("logs") / f"g1_arm_dds_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    ChannelFactoryInitialize(args.domain, args.network_interface or None)

    events: queue.Queue[dict] = queue.Queue(maxsize=50000)
    dropped = 0
    dropped_lock = threading.Lock()

    def enqueue(event: dict) -> None:
        nonlocal dropped
        try:
            events.put_nowait(event)
        except queue.Full:
            with dropped_lock:
                dropped += 1

    subscribers = []
    for topic in ("rt/lowcmd", "rt/arm_sdk"):
        subscriber = ChannelSubscriber(topic, LowCmd_)
        subscriber.Init(lambda msg, topic=topic: enqueue(_command_event(msg, topic)), 10)
        subscribers.append(subscriber)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(lambda msg: enqueue(_state_event(msg)), 10)
    subscribers.append(subscriber)

    counts = {"command": 0, "state": 0}
    latest_command = None
    latest_state = None
    start = time.monotonic()
    next_report = start + 1.0
    print(f"Recording domain {args.domain} to {output.resolve()}; Ctrl+C stops.", flush=True)
    try:
        with output.open("w", encoding="utf-8") as stream:
            while args.duration == 0 or time.monotonic() - start < args.duration:
                try:
                    event = events.get(timeout=0.1)
                except queue.Empty:
                    event = None
                if event is not None:
                    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                    counts[event["stream"]] += 1
                    if event["stream"] == "command":
                        latest_command = event
                    else:
                        latest_state = event
                now = time.monotonic()
                if now >= next_report:
                    with dropped_lock:
                        dropped_now = dropped
                    print(
                        f"cmd={counts['command']} state={counts['state']} dropped={dropped_now} "
                        f"max|tau_cmd|={_format(_maximum(latest_command, 'tau'))} Nm "
                        f"max|tau_est|={_format(_maximum(latest_state, 'tau_est'))} Nm "
                        f"max|q_cmd-q|={_format(_position_error(latest_command, latest_state))} rad",
                        flush=True,
                    )
                    stream.flush()
                    next_report = now + 1.0
    except KeyboardInterrupt:
        pass
    finally:
        for subscriber in subscribers:
            subscriber.Close()
    with dropped_lock:
        dropped_now = dropped
    print(
        f"Saved {counts['command']} command and {counts['state']} state packets to "
        f"{output.resolve()} (dropped={dropped_now}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
