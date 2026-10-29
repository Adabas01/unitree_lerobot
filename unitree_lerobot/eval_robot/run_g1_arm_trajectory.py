"""Preview or explicitly execute one fixed-torso G1-D arm experiment.

Preview imports neither DDS nor the robot controller. --execute publishes motor
commands; run only with the robot's normal operating mode and stop control ready.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np

from arm_comparison import (JOINTS, KD, KP, URDF, Trajectory, clip_target,
                            load_events, positive, validate_pose)


class EventWriter:
    """Keep disk IO off the control loop; any overflow invalidates the run."""
    def __init__(self, path):
        self.file = Path(path).open("x")
        self.queue = queue.Queue(maxsize=20000)
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                event = self.queue.get()
                if event is None:
                    break
                self.file.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
                if self.queue.empty():
                    self.file.flush()
        except Exception as exc:
            self.error = exc
        finally:
            self.file.close()

    def add(self, event):
        if self.error:
            raise RuntimeError("Event writer failed") from self.error
        try:
            self.queue.put_nowait(event)
        except queue.Full as exc:
            self.error = exc
            raise RuntimeError("Event queue overflow; recording is invalid") from exc

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=5)
        if self.thread.is_alive() or self.error:
            raise RuntimeError("Event writer failed to drain") from self.error


def gravity_model(args):
    import pinocchio as pin
    full = pin.buildModelFromUrdf(str(URDF))
    reference = pin.neutral(full)
    positions = {"Yaw_Joint": args.torso_pitch, "LZ_mt_Joint": args.lift_positions[0],
                 "LZ_it_Joint": args.lift_positions[1], "torso_Joint": args.waist_yaw}
    positions.update({f"{side}_dex1_finger_joint_{i}": args.gripper_position
                      for side in ("left", "right") for i in (1, 2)})
    for name, q in positions.items():
        reference[full.joints[full.getJointId(name)].idx_q] = q
    names = {f"{j}_joint" for j in JOINTS}
    model = pin.buildReducedModel(full, [i for i, name in enumerate(full.names) if i and name not in names], reference)
    if tuple(model.names[1:]) != tuple(f"{j}_joint" for j in JOINTS):
        raise ValueError("Gravity model joint order mismatch")
    data = model.createData()
    return lambda q: np.asarray(pin.computeGeneralizedGravity(model, data, q)).copy()


def execute(args, plan, output):
    # All robot imports are deliberately below the --execute gate.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from record_g1_arm_dds import _command_event, _state_event

    ChannelFactoryInitialize(args.domain, args.network_interface)
    writer = EventWriter(output / "events.jsonl")
    lock = threading.Lock()
    latest = None
    last_tick = None
    callback_error = None
    subscribers = []
    publisher = None
    competing = []
    publishing = False
    sent_crcs = deque(maxlen=1024)
    status, reason = "aborted", "initialization did not complete"
    dt = 1 / 250.
    crc = CRC()

    def state_callback(msg):
        nonlocal latest, last_tick, callback_error
        try:
            event = _state_event(msg)
            writer.add(event)
            if not event["crc_valid"]:
                raise ValueError("Invalid lowstate CRC")
            q = np.array([motor.q for motor in msg.motor_state[:29]])
            dq = np.array([motor.dq for motor in msg.motor_state[:29]])
            if not np.isfinite(q).all() or not np.isfinite(dq).all():
                raise ValueError("Nonfinite lowstate")
            with lock:
                if msg.tick != last_tick:
                    latest = (event["time_monotonic_ns"], q, dq, int(msg.mode_machine))
                    last_tick = msg.tick
        except Exception as exc:
            callback_error = exc

    def snapshot():
        if callback_error:
            raise RuntimeError("State recording failed") from callback_error
        if writer.error:
            raise RuntimeError("Recording failed") from writer.error
        with lock:
            state = latest
        if state is None or (time.monotonic_ns() - state[0]) / 1e9 > args.state_timeout:
            raise RuntimeError("Missing or stale lowstate; stopping publication")
        return state

    def observe_command(msg, topic):
        nonlocal callback_error
        if not publishing:
            competing.append(topic)
        else:
            with lock:
                own = topic == args.command_topic and msg.crc in sent_crcs
            if not own:
                callback_error = RuntimeError(f"Unexpected command publisher on {topic}")

    try:
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscribers.append(subscriber)
        subscriber.Init(state_callback)  # no intermediate SDK queue dropping older packets
        for topic in ("rt/lowcmd", "rt/arm_sdk"):
            subscriber = ChannelSubscriber(topic, LowCmd_)
            subscribers.append(subscriber)
            subscriber.Init(lambda msg, topic=topic: observe_command(msg, topic))
        deadline = time.monotonic() + 8
        while latest is None and time.monotonic() < deadline and callback_error is None:
            time.sleep(.02)
        snapshot()
        time.sleep(.5)
        _, initial_q, initial_dq, mode_machine = snapshot()
        if competing:
            raise RuntimeError(f"Another command publisher is active on {set(competing)}; stop it first")
        if np.max(abs(initial_dq[15:29])) > .1:
            raise RuntimeError("Arms must be stationary before starting")
        center = initial_q[15:29].copy()
        validate_pose(plan, center, args.limit_margin)
        args.waist_yaw = float(initial_q[12])
        gravity = gravity_model(args) if args.gravity_feedforward else lambda q: np.zeros(14)
        # Warm up before starting the timed loop.
        gravity(center)
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr, msg.mode_machine = 0, mode_machine
        for i in range(29):
            motor = msg.motor_cmd[i]
            motor.mode, motor.q, motor.dq, motor.tau = 1, float(initial_q[i]), 0., 0.
            motor.kp, motor.kd = (80., 3.) if i in (4, 5, 10, 11) else (300., 3.)
        for j, i in enumerate(range(15, 29)):
            msg.motor_cmd[i].kp, msg.motor_cmd[i].kd = float(KP[j]), float(KD[j])
        if args.command_topic == "rt/arm_sdk":
            msg.motor_cmd[29].q = 1.
        publisher = ChannelPublisher(args.command_topic, LowCmd_)
        publisher.Init()
        if competing:
            raise RuntimeError("Another command publisher became active during startup")
        _, current_q, current_dq, _ = snapshot()
        if np.max(abs(current_q[12:29] - initial_q[12:29])) > .02 or np.max(abs(current_dq[15:29])) > .1:
            raise RuntimeError("Starting pose moved during initialization; retry from a stationary pose")
        origin = time.monotonic_ns()
        manifest = dict(schema_version=1, stream="metadata", joint_names=list(JOINTS),
                        trajectory=asdict(plan), trajectory_origin_ns=origin, domain=args.domain,
                        command_topic=args.command_topic, initial_arm_q=center.tolist(),
                        fixed_configuration=dict(torso_pitch=args.torso_pitch, torso_yaw=args.waist_yaw,
                                                 lift_positions=args.lift_positions, gripper_position=args.gripper_position),
                        gravity_feedforward=args.gravity_feedforward, gravity_model=str(URDF),
                        gravity_model_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
                        publish_hz=250., target_hz=args.target_rate,
                        timestamp_semantics="host transmit attempt and host state callback receipt",
                        notes=args.notes, arguments=vars(args).copy())
        manifest["arguments"]["output"] = str(args.output)
        manifest["software"] = {}
        for file in (Path(__file__), Path(__file__).with_name("arm_comparison.py")):
            manifest["software"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        writer.add(manifest)
        publishing = True
        target, tau = center.copy(), gravity(center)
        next_target, next_tick, previous_phase = 0., time.monotonic(), None
        print(f"Executing {plan.kind} on {plan.joint} for {plan.duration:.1f}s; Ctrl+C aborts.", flush=True)
        while True:
            now_ns = time.monotonic_ns()
            t = (now_ns - origin) / 1e9
            if t > plan.duration:
                break
            _, q, dq, machine = snapshot()
            if machine != mode_machine:
                raise RuntimeError("Robot mode changed")
            if abs(q[12] - initial_q[12]) > args.max_torso_drift:
                raise RuntimeError("Torso yaw moved beyond fixed-torso tolerance")
            if np.max(abs(dq[15:29])) > args.max_measured_speed:
                raise RuntimeError("Measured arm speed exceeded bound")
            if np.max(abs(q[15:29] - target)) > args.max_tracking_error:
                raise RuntimeError("Arm tracking error exceeded bound")
            if t >= next_target:
                displacement, phase = plan.sample(t)
                target = center.copy()
                target[JOINTS.index(plan.joint)] += displacement
                tau = gravity(q[15:29])
                if not np.isfinite(tau).all():
                    raise RuntimeError("Nonfinite feedforward")
                writer.add(dict(stream="desired", time_monotonic_ns=now_ns, q=target.tolist(), phase=phase))
                if phase != previous_phase:
                    print(f"{t:7.3f}s  {phase}", flush=True)
                    previous_phase = phase
                next_target = t + 1 / args.target_rate
            clipped = clip_target(target, q[15:29], dt)
            for j, i in enumerate(range(15, 29)):
                msg.motor_cmd[i].q = float(clipped[j])
                msg.motor_cmd[i].tau = float(tau[j])
            msg.crc = crc.Crc(msg)
            event = _command_event(msg, args.command_topic)
            with lock:
                sent_crcs.append(msg.crc)
            if not publisher.Write(msg):
                raise RuntimeError("DDS write failed")
            writer.add(event)
            next_tick += dt
            late = time.monotonic() - next_tick
            if late > args.state_timeout:
                raise RuntimeError("Control loop missed its deadline by too much")
            # Do not burst old commands when late; preserve actual transmit timestamps.
            if late > dt:
                next_tick = time.monotonic()
            time.sleep(max(0., next_tick - time.monotonic()))
        snapshot()
        status, reason = "complete", "trajectory and final hold finished"
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # No background publisher survives this function. The robot's operating
        # mode/watchdog determines behavior after publication stops.
        if publisher is not None:
            publisher.Close()
        for subscriber in subscribers:
            subscriber.Close()
        try:
            writer.add(dict(stream="end", status=status, reason=reason, time_monotonic_ns=time.monotonic_ns()))
        finally:
            writer.close()
        (output / "result.json").write_text(json.dumps(dict(status=status, reason=reason), indent=2) + "\n")
    trial = load_events(output / "events.jsonl")
    trial.save(output / "trial.npz")
    print(f"Saved {output / 'trial.npz'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--joint", choices=JOINTS, default="left_elbow")
    parser.add_argument("--trajectory", choices=("hold", "point", "sine"), default="sine")
    parser.add_argument("--amplitude", type=float, default=.05, help="Radians; signed for point trajectory")
    parser.add_argument("--sine-frequency", type=float, default=.25, help="Hz")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--ramp", type=float, default=2.)
    parser.add_argument("--hold", type=float, default=5.)
    parser.add_argument("--execute", action="store_true", help="Enable real/sim DDS publication")
    parser.add_argument("--domain", choices=(0, 1), type=int, default=0)
    parser.add_argument("--network-interface", default="")
    parser.add_argument("--command-topic", choices=("rt/lowcmd", "rt/arm_sdk"), default="rt/lowcmd")
    parser.add_argument("--gravity-feedforward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-rate", type=float, default=30.)
    parser.add_argument("--limit-margin", type=float, default=.05)
    parser.add_argument("--max-target-speed", type=float, default=.3)
    parser.add_argument("--max-target-acceleration", type=float, default=1.)
    parser.add_argument("--max-measured-speed", type=float, default=1.5)
    parser.add_argument("--max-tracking-error", type=float, default=.25)
    parser.add_argument("--max-torso-drift", type=float, default=.03)
    parser.add_argument("--state-timeout", type=float, default=.1)
    parser.add_argument("--torso-pitch", type=float, default=0., help="Fixed physical torso pitch in radians")
    parser.add_argument("--lift-positions", type=float, nargs=2, default=[0., 0.])
    parser.add_argument("--gripper-position", type=float, default=.0245, help="Fixed finger position in metres; .0245=open")
    parser.add_argument("--notes", default="", help="Hardware attachments, pose, and run notes")
    args = parser.parse_args()
    plan = Trajectory(args.joint, args.trajectory, args.amplitude, args.sine_frequency, args.cycles, args.ramp, args.hold)
    for name in ("target_rate", "max_target_speed", "max_target_acceleration", "max_measured_speed",
                 "max_tracking_error", "max_torso_drift", "state_timeout"):
        positive(getattr(args, name), name)
    if args.target_rate > 250:
        parser.error("target rate cannot exceed 250 Hz publication")
    positive(args.limit_margin, "limit margin", allow_zero=True)
    if plan.kind == "sine" and plan.frequency > args.target_rate / 10:
        parser.error("Sine frequency needs at least 10 target updates per cycle")
    if not np.isfinite([args.torso_pitch, *args.lift_positions, args.gripper_position]).all():
        parser.error("Fixed configuration must be finite")
    if not -.04363 <= args.torso_pitch <= 2.3562 or not all(0 <= q <= .21 for q in args.lift_positions):
        parser.error("Torso/lift configuration is outside model limits")
    if not -.02 <= args.gripper_position <= .0245:
        parser.error("Gripper position must be in [-.02, .0245] metres")
    if args.execute and not args.network_interface:
        parser.error("--execute requires an explicit --network-interface")
    t, offset, velocity, acceleration = plan.preview()
    if np.max(abs(velocity)) > args.max_target_speed or np.max(abs(acceleration)) > args.max_target_acceleration:
        parser.error("Trajectory exceeds target speed/acceleration bounds; reduce amplitude/frequency or increase ramp")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "trajectory.json").write_text(json.dumps(asdict(plan), indent=2) + "\n")
    np.savetxt(args.output / "preview.csv", np.column_stack((t, offset, velocity, acceleration)), delimiter=",",
               header="time_s,offset_rad,velocity_rad_s,acceleration_rad_s2", comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for ax, values, label in zip(axes, (offset, velocity, acceleration), ("offset [rad]", "speed [rad/s]", "acceleration [rad/s²]")):
        ax.plot(t, values)
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{plan.joint}: {plan.kind}; offset around measured starting pose")
    fig.tight_layout()
    fig.savefig(args.output / "preview.png", dpi=130)
    plt.close(fig)
    print(f"Duration={plan.duration:.1f}s; peak speed={max(abs(velocity)):.4f} rad/s; "
          f"peak acceleration={max(abs(acceleration)):.4f} rad/s²", flush=True)
    if args.execute:
        execute(args, plan, args.output)
    else:
        print(f"Preview only: {args.output}; no DDS initialized or commands sent.")


if __name__ == "__main__":
    main()
