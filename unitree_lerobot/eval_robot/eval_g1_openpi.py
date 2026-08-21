"""Run an OpenPI websocket policy on a real Unitree G1 Dex1 setup.

This bridge intentionally depends only on ``openpi-client`` on the robot side.
Run the full OpenPI/JAX policy server from ``openpi-g1d`` in a separate terminal
or on a GPU server, then use this script from the Unitree/LeRobot environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
import logging
from multiprocessing.sharedctypes import SynchronizedArray
import os
import time
from typing import Any

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from unitree_lerobot.eval_robot.make_robot import process_images_and_observations
from unitree_lerobot.eval_robot.make_robot import setup_image_client
from unitree_lerobot.eval_robot.make_robot import setup_robot_interface
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger
from unitree_lerobot.eval_robot.utils.rerun_visualizer import visualization_data
from unitree_lerobot.eval_robot.utils.utils import to_scalar

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except ImportError as exc:  # pragma: no cover - this is a deployment hint.
    raise ImportError(
        "openpi-client is required in the robot environment. Install it with:\n"
        "  pip install -e ~/adam_unitree/openpi-g1d/packages/openpi-client"
    ) from exc


LOGGER = logging.getLogger(__name__)

TASK_PROMPT = "Sort the cubes into the brown cardboard box and the cylinders into the green plastic bin."
UNITREE_TO_OPENPI = np.array([0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15])
OPENPI_TO_UNITREE = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7, 15])


@dataclass(frozen=True)
class InferenceResult:
    chunk: np.ndarray
    client_ms: float
    server_ms: float | None
    model_ms: float | None
    prev_total_ms: float | None


@dataclass
class Args:
    policy_host: str = "localhost"
    policy_port: int = 8000
    image_host: str = "192.168.212.34"
    network_interface: str = ""
    frequency: float = 30.0
    dds_timeout: float = 8.0
    prompt: str = TASK_PROMPT
    arm: str = "G1_29"
    ee: str = "dex1"
    motion: bool = False
    sim: bool = False
    visualization: bool = False
    camera_only: bool = False
    home_before_start: bool = False
    home_duration: float = 4.0
    home_max_step: float = 0.015
    home_tolerance: float = 0.05
    home_timeout: float = 10.0
    home_open_grippers: bool = True
    home_gripper_open: float = 1.0
    home_gripper_max_step: float = 0.03
    execute_actions: bool = False
    execute_grippers: bool = False
    max_steps: int = 0
    execute_chunk_steps: int = 1
    async_policy: bool = True
    prefetch_chunk_steps: int = 4
    action_mode: str = "limited"
    max_arm_step: float = 0.05
    max_gripper_step: float = 0.20
    arm_smoothing: float = 0.0
    gripper_smoothing: float = 0.0
    gripper_scale: float = 5.4
    policy_layout: str = "interleaved"
    print_every: int = 10
    profile_timing: bool = False
    profile_every: int = 30
    profile_csv: str = ""


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default=Args.policy_host)
    parser.add_argument("--policy-port", type=int, default=Args.policy_port)
    parser.add_argument("--image-host", default=Args.image_host)
    parser.add_argument("--network-interface", default=Args.network_interface)
    parser.add_argument("--frequency", type=float, default=Args.frequency)
    parser.add_argument("--dds-timeout", type=float, default=Args.dds_timeout)
    parser.add_argument("--prompt", default=Args.prompt)
    parser.add_argument("--arm", default=Args.arm)
    parser.add_argument("--ee", default=Args.ee)
    parser.add_argument("--motion", type=_parse_bool, default=Args.motion)
    parser.add_argument("--sim", type=_parse_bool, default=Args.sim)
    parser.add_argument("--visualization", type=_parse_bool, default=Args.visualization)
    parser.add_argument("--camera-only", type=_parse_bool, default=Args.camera_only)
    parser.add_argument("--home-before-start", type=_parse_bool, default=Args.home_before_start)
    parser.add_argument("--home-duration", type=float, default=Args.home_duration)
    parser.add_argument("--home-max-step", type=float, default=Args.home_max_step)
    parser.add_argument("--home-tolerance", type=float, default=Args.home_tolerance)
    parser.add_argument("--home-timeout", type=float, default=Args.home_timeout)
    parser.add_argument("--home-open-grippers", type=_parse_bool, default=Args.home_open_grippers)
    parser.add_argument("--home-gripper-open", type=float, default=Args.home_gripper_open)
    parser.add_argument("--home-gripper-max-step", type=float, default=Args.home_gripper_max_step)
    parser.add_argument("--execute-actions", type=_parse_bool, default=Args.execute_actions)
    parser.add_argument("--execute-grippers", type=_parse_bool, default=Args.execute_grippers)
    parser.add_argument("--max-steps", type=int, default=Args.max_steps)
    parser.add_argument("--execute-chunk-steps", type=int, default=Args.execute_chunk_steps)
    parser.add_argument("--async-policy", type=_parse_bool, default=Args.async_policy)
    parser.add_argument("--prefetch-chunk-steps", type=int, default=Args.prefetch_chunk_steps)
    parser.add_argument("--action-mode", choices=("limited", "direct"), default=Args.action_mode)
    parser.add_argument("--max-arm-step", type=float, default=Args.max_arm_step)
    parser.add_argument("--max-gripper-step", type=float, default=Args.max_gripper_step)
    parser.add_argument("--arm-smoothing", type=float, default=Args.arm_smoothing)
    parser.add_argument("--gripper-smoothing", type=float, default=Args.gripper_smoothing)
    parser.add_argument("--gripper-scale", type=float, default=Args.gripper_scale)
    parser.add_argument("--policy-layout", choices=("interleaved", "unitree"), default=Args.policy_layout)
    parser.add_argument("--print-every", type=int, default=Args.print_every)
    parser.add_argument("--profile-timing", type=_parse_bool, default=Args.profile_timing)
    parser.add_argument("--profile-every", type=int, default=Args.profile_every)
    parser.add_argument("--profile-csv", default=Args.profile_csv)
    return Args(**vars(parser.parse_args()))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _available_interfaces() -> list[str]:
    try:
        return sorted(os.listdir("/sys/class/net"))
    except OSError:
        return []


def _check_network_interface(network_interface: str) -> None:
    if not network_interface:
        return
    available = _available_interfaces()
    if available and network_interface not in available:
        raise RuntimeError(
            f"Network interface '{network_interface}' is not available. "
            f"Available interfaces: {', '.join(available)}"
        )


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., ::-1])


def _camera_observation_from_client(img_client: Any, camera_config: dict[str, Any]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}

    if camera_config["head_camera"]["enable_zmq"]:
        head_img = img_client.get_head_frame()
        if head_img is not None and head_img.bgr is not None:
            width = int(camera_config["head_camera"]["image_shape"][1])
            observation["observation.images.cam_left_high"] = _bgr_to_rgb(head_img.bgr[:, : width // 2])

    if camera_config["left_wrist_camera"]["enable_zmq"]:
        left_wrist = img_client.get_left_wrist_frame()
        if left_wrist is not None and left_wrist.bgr is not None:
            observation["observation.images.cam_left_wrist"] = _bgr_to_rgb(left_wrist.bgr)

    if camera_config["right_wrist_camera"]["enable_zmq"]:
        right_wrist = img_client.get_right_wrist_frame()
        if right_wrist is not None and right_wrist.bgr is not None:
            observation["observation.images.cam_right_wrist"] = _bgr_to_rgb(right_wrist.bgr)

    return observation


def _require_rgb_image(observation: dict[str, Any], key: str) -> np.ndarray:
    image = observation.get(key)
    if image is None:
        raise RuntimeError(f"Missing image '{key}'. Check teleimager config and camera stream.")
    image = _as_numpy(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"Expected HWC RGB image for '{key}', got shape {image.shape}.")
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))


def _unitree_state_to_policy_state(unitree_state: np.ndarray, policy_layout: str) -> np.ndarray:
    if policy_layout == "unitree":
        return unitree_state.astype(np.float32, copy=False)
    return unitree_state[UNITREE_TO_OPENPI].astype(np.float32, copy=False)


def _policy_action_to_unitree_action(policy_action: np.ndarray, policy_layout: str) -> np.ndarray:
    if policy_layout == "unitree":
        return policy_action.astype(np.float64, copy=False)
    return policy_action[OPENPI_TO_UNITREE].astype(np.float64, copy=False)


def _infer_action_chunk(policy: Any, policy_observation: dict[str, Any]) -> InferenceResult:
    start = time.perf_counter()
    result = policy.infer(policy_observation)
    infer_ms = 1000.0 * (time.perf_counter() - start)
    chunk = np.asarray(result["actions"], dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 16:
        raise RuntimeError(f"Expected policy actions with shape [T, 16], got {chunk.shape}.")
    if not np.all(np.isfinite(chunk)):
        raise RuntimeError("Policy returned non-finite actions.")
    server_timing = result.get("server_timing", {})
    policy_timing = result.get("policy_timing", {})
    return InferenceResult(
        chunk=chunk,
        client_ms=infer_ms,
        server_ms=server_timing.get("infer_ms"),
        model_ms=policy_timing.get("infer_ms"),
        prev_total_ms=server_timing.get("prev_total_ms"),
    )


def _optional_ms(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _build_policy_observation(
    observation: dict[str, Any],
    current_arm_q: np.ndarray,
    gripper_state_raw: np.ndarray,
    args: Args,
) -> dict[str, Any]:
    if current_arm_q is None:
        raise RuntimeError("Arm state is unavailable; check rt/lowstate DDS.")
    if current_arm_q.shape[0] != 14:
        raise RuntimeError(f"Expected 14 arm joints, got {current_arm_q.shape}.")

    gripper_state_norm = np.clip(gripper_state_raw / args.gripper_scale, 0.0, 1.0)
    unitree_state = np.concatenate([current_arm_q, gripper_state_norm], axis=0)
    policy_state = _unitree_state_to_policy_state(unitree_state, args.policy_layout)

    return {
        "observation/image_high": _require_rgb_image(observation, "observation.images.cam_left_high"),
        "observation/image_left_wrist": _require_rgb_image(observation, "observation.images.cam_left_wrist"),
        "observation/image_right_wrist": _require_rgb_image(observation, "observation.images.cam_right_wrist"),
        "observation/state": policy_state,
        "prompt": args.prompt,
    }


def _read_gripper_state(ee_shared_mem: dict[str, Any], ee_dof: int) -> np.ndarray:
    if ee_dof == 0:
        return np.array([], dtype=np.float64)
    with ee_shared_mem["lock"]:
        state = np.array(ee_shared_mem["state"][:], dtype=np.float64)
    if state.shape[0] < 2 * ee_dof:
        raise RuntimeError(f"Expected at least {2 * ee_dof} gripper state values, got {state.shape}.")
    return state[: 2 * ee_dof]


def _clip_action(
    action_unitree: np.ndarray,
    current_arm_q: np.ndarray,
    gripper_state_raw: np.ndarray,
    args: Args,
) -> tuple[np.ndarray, np.ndarray]:
    arm_target = action_unitree[:14]
    arm_target = current_arm_q + np.clip(arm_target - current_arm_q, -args.max_arm_step, args.max_arm_step)

    gripper_target_norm = np.clip(action_unitree[14:16], 0.0, 1.0)
    gripper_target_raw = gripper_target_norm * args.gripper_scale
    gripper_target_raw = gripper_state_raw + np.clip(
        gripper_target_raw - gripper_state_raw,
        -args.max_gripper_step * args.gripper_scale,
        args.max_gripper_step * args.gripper_scale,
    )
    gripper_target_raw = np.clip(gripper_target_raw, 0.0, args.gripper_scale)

    return arm_target, gripper_target_raw


def _smooth_target(target: np.ndarray, previous: np.ndarray | None, smoothing: float) -> np.ndarray:
    if previous is None or smoothing <= 0.0:
        return target
    return (smoothing * previous) + ((1.0 - smoothing) * target)


def _write_gripper_targets(ee_shared_mem: dict[str, Any], left: float, right: float) -> None:
    if isinstance(ee_shared_mem["left"], SynchronizedArray):
        ee_shared_mem["left"][:] = [left]
        ee_shared_mem["right"][:] = [right]
    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
        ee_shared_mem["left"].value = to_scalar(left)
        ee_shared_mem["right"].value = to_scalar(right)
    else:
        raise RuntimeError("Unsupported gripper shared-memory type.")


def _move_arms_home(arm_ctrl: Any, arm_ik: Any, ee_shared_mem: dict[str, Any], args: Args) -> None:
    current_q = arm_ctrl.get_current_dual_arm_q().astype(np.float64, copy=False)
    home_q = np.zeros_like(current_q)
    previous_target = current_q.copy()
    gripper_target_raw = np.clip(args.home_gripper_open, 0.0, 1.0) * args.gripper_scale
    loop_dt = 1.0 / max(args.frequency, 1.0)
    ramp_steps = max(1, int(args.home_duration / loop_dt))

    LOGGER.info(
        "Moving arms home before start: duration=%.1fs max_step=%.4f tolerance=%.3f open_grippers=%s",
        args.home_duration,
        args.home_max_step,
        args.home_tolerance,
        args.home_open_grippers,
    )

    for step in range(ramp_steps):
        loop_start = time.perf_counter()
        alpha = (step + 1) / ramp_steps
        planned_target = current_q + alpha * (home_q - current_q)
        target = previous_target + np.clip(
            planned_target - previous_target,
            -args.home_max_step,
            args.home_max_step,
        )
        tau = arm_ik.solve_tau(target)
        arm_ctrl.ctrl_dual_arm(target, tau)
        if args.home_open_grippers:
            current_gripper_raw = _read_gripper_state(ee_shared_mem, 1)
            next_gripper_raw = current_gripper_raw + np.clip(
                gripper_target_raw - current_gripper_raw,
                -args.home_gripper_max_step * args.gripper_scale,
                args.home_gripper_max_step * args.gripper_scale,
            )
            _write_gripper_targets(ee_shared_mem, next_gripper_raw[0], next_gripper_raw[1])
        previous_target = target
        time.sleep(max(0.0, loop_dt - (time.perf_counter() - loop_start)))

    tau = arm_ik.solve_tau(home_q)
    arm_ctrl.ctrl_dual_arm(home_q, tau)
    if args.home_open_grippers:
        _write_gripper_targets(ee_shared_mem, gripper_target_raw, gripper_target_raw)

    wait_start = time.monotonic()
    while time.monotonic() - wait_start < args.home_timeout:
        current_q = arm_ctrl.get_current_dual_arm_q()
        max_error = float(np.max(np.abs(current_q - home_q)))
        if max_error <= args.home_tolerance:
            LOGGER.info("Arms reached home position; max_error=%.4f", max_error)
            return
        time.sleep(0.05)

    current_q = arm_ctrl.get_current_dual_arm_q()
    LOGGER.warning(
        "Timed out waiting for arms to reach home; max_error=%.4f. Continuing to start prompt.",
        float(np.max(np.abs(current_q - home_q))),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    if args.ee.lower() != "dex1":
        raise ValueError("This OpenPI G1 bridge currently supports --ee=dex1 only.")
    if args.execute_chunk_steps < 1:
        raise ValueError("--execute-chunk-steps must be >= 1.")
    if args.prefetch_chunk_steps < 0:
        raise ValueError("--prefetch-chunk-steps must be >= 0.")
    if args.gripper_scale <= 0:
        raise ValueError("--gripper-scale must be positive.")
    if args.home_duration <= 0:
        raise ValueError("--home-duration must be positive.")
    if args.home_max_step <= 0:
        raise ValueError("--home-max-step must be positive.")
    if args.home_tolerance <= 0:
        raise ValueError("--home-tolerance must be positive.")
    if args.home_timeout <= 0:
        raise ValueError("--home-timeout must be positive.")
    if not 0.0 <= args.home_gripper_open <= 1.0:
        raise ValueError("--home-gripper-open must be in [0, 1].")
    if args.home_gripper_max_step <= 0:
        raise ValueError("--home-gripper-max-step must be positive.")
    if not 0.0 <= args.arm_smoothing < 1.0:
        raise ValueError("--arm-smoothing must be in [0, 1).")
    if not 0.0 <= args.gripper_smoothing < 1.0:
        raise ValueError("--gripper-smoothing must be in [0, 1).")
    if args.profile_every < 1:
        raise ValueError("--profile-every must be >= 1.")

    LOGGER.info("Connecting to OpenPI policy server at %s:%s", args.policy_host, args.policy_port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    LOGGER.info("Policy server metadata: %s", policy.get_server_metadata())

    image_client = None
    executor = None
    profile_file = None
    profile_writer = None
    rerun_logger = RerunLogger() if args.visualization else None
    try:
        if args.profile_csv:
            profile_file = open(args.profile_csv, "w", newline="", encoding="utf-8")
            profile_fields = [
                "step",
                "chunk_index",
                "active_chunk_steps",
                "chunk_boundary",
                "extended_chunk",
                "prefetch_submitted",
                "prefetch_ready",
                "loop_ms",
                "obs_ms",
                "gripper_ms",
                "build_ms",
                "policy_wait_ms",
                "client_infer_ms",
                "server_infer_ms",
                "model_infer_ms",
                "server_prev_total_ms",
                "action_ms",
                "cmd_ms",
                "viz_ms",
                "sleep_ms",
                "overrun_ms",
                "arm_delta_max",
                "policy_action_jump_max",
                "arm_cmd_jump_max",
                "gripper_cmd_jump_norm",
            ]
            profile_writer = csv.DictWriter(profile_file, fieldnames=profile_fields)
            profile_writer.writeheader()
            LOGGER.info("Writing timing profile CSV to %s", args.profile_csv)

        LOGGER.info("Connecting to teleimager at %s", args.image_host)
        image_client, camera_config = setup_image_client(args)

        if args.camera_only:
            LOGGER.info("camera_only=true; skipping DDS and robot controllers.")
            loop_dt = 1.0 / args.frequency
            step_count = args.max_steps if args.max_steps > 0 else 50
            for step_idx in range(step_count):
                loop_start = time.perf_counter()
                observation = _camera_observation_from_client(image_client, camera_config)
                policy_obs = _build_policy_observation(
                    observation,
                    np.zeros(14, dtype=np.float64),
                    np.zeros(2, dtype=np.float64),
                    args,
                )
                inference = _infer_action_chunk(policy, policy_obs)
                actions = inference.chunk
                LOGGER.info(
                    "camera step=%d action_shape=%s infer_ms client/server/model=%.1f/%.1f/%.1f "
                    "first_gripper_norm=(%.3f, %.3f)",
                    step_idx,
                    actions.shape,
                    inference.client_ms,
                    _optional_ms(inference.server_ms),
                    _optional_ms(inference.model_ms),
                    float(actions[0, 7] if args.policy_layout == "interleaved" else actions[0, 14]),
                    float(actions[0, 15]),
                )
                time.sleep(max(0.0, loop_dt - (time.perf_counter() - loop_start)))
            return

        _check_network_interface(args.network_interface)
        domain_id = 1 if args.sim else 0
        ChannelFactoryInitialize(domain_id, args.network_interface or None)
        robot_interface = setup_robot_interface(args)
        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]
        ee_dof = int(robot_interface["ee_dof"])

        if args.home_before_start:
            _move_arms_home(arm_ctrl, arm_ik, ee_shared_mem, args)

        LOGGER.info(
            "execute_actions=%s execute_grippers=%s async_policy=%s prefetch_chunk_steps=%d. "
            "action_mode=%s profile_timing=%s. The script will query policy actions either way.",
            args.execute_actions,
            args.execute_grippers,
            args.async_policy,
            args.prefetch_chunk_steps,
            args.action_mode,
            args.profile_timing,
        )
        user_input = input("Enter 's' to start OpenPI evaluation loop: ")
        if user_input.lower() != "s":
            LOGGER.info("Start signal not received; exiting.")
            return

        if args.async_policy:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        chunk = None
        chunk_index = 0
        prefetch_future = None
        last_inference: InferenceResult | None = None
        step_idx = 0
        loop_dt = 1.0 / args.frequency
        previous_arm_target = None
        previous_gripper_target_raw = None
        previous_policy_action_unitree = None

        while args.max_steps <= 0 or step_idx < args.max_steps:
            loop_start = time.perf_counter()
            timing = {
                "obs_ms": 0.0,
                "gripper_ms": 0.0,
                "build_ms": 0.0,
                "policy_wait_ms": 0.0,
                "action_ms": 0.0,
                "cmd_ms": 0.0,
                "viz_ms": 0.0,
                "sleep_ms": 0.0,
                "overrun_ms": 0.0,
            }
            prefetch_submitted = False
            prefetch_ready = False
            chunk_boundary = False
            using_extended_chunk = False

            stage_start = time.perf_counter()
            observation, current_arm_q = process_images_and_observations(image_client, camera_config, arm_ctrl)
            current_arm_q = _as_numpy(current_arm_q).astype(np.float64, copy=False)
            timing["obs_ms"] = 1000.0 * (time.perf_counter() - stage_start)

            stage_start = time.perf_counter()
            gripper_state_raw = _read_gripper_state(ee_shared_mem, ee_dof)
            timing["gripper_ms"] = 1000.0 * (time.perf_counter() - stage_start)

            if chunk is None:
                stage_start = time.perf_counter()
                policy_obs = _build_policy_observation(observation, current_arm_q, gripper_state_raw, args)
                timing["build_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                stage_start = time.perf_counter()
                last_inference = _infer_action_chunk(policy, policy_obs)
                timing["policy_wait_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                chunk = last_inference.chunk
                chunk_index = 0

            active_chunk_steps = min(args.execute_chunk_steps, len(chunk))
            prefetch_at = max(0, active_chunk_steps - args.prefetch_chunk_steps)
            if args.async_policy and prefetch_future is None and chunk_index >= prefetch_at:
                stage_start = time.perf_counter()
                policy_obs = _build_policy_observation(observation, current_arm_q, gripper_state_raw, args)
                timing["build_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                prefetch_future = executor.submit(_infer_action_chunk, policy, policy_obs)
                prefetch_submitted = True

            if chunk_index >= active_chunk_steps:
                chunk_boundary = True
                if args.async_policy:
                    if prefetch_future is not None and prefetch_future.done():
                        prefetch_ready = True
                        stage_start = time.perf_counter()
                        last_inference = prefetch_future.result()
                        timing["policy_wait_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                        chunk = last_inference.chunk
                        chunk_index = 0
                        prefetch_future = None
                    elif chunk_index < len(chunk):
                        using_extended_chunk = True
                    else:
                        LOGGER.warning("Waiting for OpenPI inference at the end of the action chunk.")
                        if prefetch_future is None:
                            stage_start = time.perf_counter()
                            policy_obs = _build_policy_observation(observation, current_arm_q, gripper_state_raw, args)
                            timing["build_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                            stage_start = time.perf_counter()
                            last_inference = _infer_action_chunk(policy, policy_obs)
                            timing["policy_wait_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                        else:
                            stage_start = time.perf_counter()
                            last_inference = prefetch_future.result()
                            timing["policy_wait_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                        chunk = last_inference.chunk
                        chunk_index = 0
                        prefetch_future = None
                else:
                    stage_start = time.perf_counter()
                    policy_obs = _build_policy_observation(observation, current_arm_q, gripper_state_raw, args)
                    timing["build_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                    stage_start = time.perf_counter()
                    last_inference = _infer_action_chunk(policy, policy_obs)
                    timing["policy_wait_ms"] += 1000.0 * (time.perf_counter() - stage_start)
                    chunk = last_inference.chunk
                    chunk_index = 0

            stage_start = time.perf_counter()
            action_unitree = _policy_action_to_unitree_action(chunk[chunk_index], args.policy_layout)
            policy_action_jump_max = (
                float("nan")
                if previous_policy_action_unitree is None
                else float(np.max(np.abs(action_unitree - previous_policy_action_unitree)))
            )
            policy_gripper_norm = np.clip(action_unitree[14:16], 0.0, 1.0)
            if args.action_mode == "direct":
                arm_target = action_unitree[:14]
                gripper_target_raw = policy_gripper_norm * args.gripper_scale
            else:
                arm_target, gripper_target_raw = _clip_action(action_unitree, current_arm_q, gripper_state_raw, args)
                arm_target = _smooth_target(arm_target, previous_arm_target, args.arm_smoothing)
                gripper_target_raw = _smooth_target(
                    gripper_target_raw,
                    previous_gripper_target_raw,
                    args.gripper_smoothing,
                )
            arm_cmd_jump_max = (
                float("nan")
                if previous_arm_target is None
                else float(np.max(np.abs(arm_target - previous_arm_target)))
            )
            gripper_cmd_jump_norm = (
                float("nan")
                if previous_gripper_target_raw is None
                else float(np.max(np.abs(gripper_target_raw - previous_gripper_target_raw)) / args.gripper_scale)
            )
            previous_arm_target = arm_target
            previous_gripper_target_raw = gripper_target_raw
            previous_policy_action_unitree = action_unitree
            timing["action_ms"] = 1000.0 * (time.perf_counter() - stage_start)

            stage_start = time.perf_counter()
            if args.execute_actions:
                tau = arm_ik.solve_tau(arm_target)
                arm_ctrl.ctrl_dual_arm(arm_target, tau)
                if args.execute_grippers:
                    _write_gripper_targets(ee_shared_mem, gripper_target_raw[0], gripper_target_raw[1])
            timing["cmd_ms"] = 1000.0 * (time.perf_counter() - stage_start)

            if args.print_every > 0 and step_idx % args.print_every == 0:
                client_infer_ms = _optional_ms(last_inference.client_ms if last_inference is not None else None)
                server_infer_ms = _optional_ms(last_inference.server_ms if last_inference is not None else None)
                model_infer_ms = _optional_ms(last_inference.model_ms if last_inference is not None else None)
                LOGGER.info(
                    "step=%d chunk=%d/%d mode=%s extended=%s infer_ms client/server/model=%.1f/%.1f/%.1f "
                    "execute=%s grippers=%s arm_delta_max=%.4f "
                    "gripper_state_norm=(%.3f, %.3f) policy_norm=(%.3f, %.3f) cmd_norm=(%.3f, %.3f)",
                    step_idx,
                    chunk_index,
                    active_chunk_steps,
                    args.action_mode,
                    using_extended_chunk,
                    client_infer_ms,
                    server_infer_ms,
                    model_infer_ms,
                    args.execute_actions,
                    args.execute_grippers,
                    float(np.max(np.abs(arm_target - current_arm_q))),
                    float(gripper_state_raw[0] / args.gripper_scale),
                    float(gripper_state_raw[1] / args.gripper_scale),
                    float(policy_gripper_norm[0]),
                    float(policy_gripper_norm[1]),
                    float(gripper_target_raw[0] / args.gripper_scale),
                    float(gripper_target_raw[1] / args.gripper_scale),
                )

            if rerun_logger is not None:
                stage_start = time.perf_counter()
                visualization_state = np.concatenate([current_arm_q, gripper_state_raw / args.gripper_scale])
                visualization_action = np.concatenate([arm_target, gripper_target_raw / args.gripper_scale])
                visualization_data(step_idx, observation, visualization_state, visualization_action, rerun_logger)
                timing["viz_ms"] = 1000.0 * (time.perf_counter() - stage_start)

            elapsed_before_sleep = time.perf_counter() - loop_start
            sleep_s = max(0.0, loop_dt - elapsed_before_sleep)
            timing["sleep_ms"] = 1000.0 * sleep_s
            timing["overrun_ms"] = max(0.0, 1000.0 * (elapsed_before_sleep - loop_dt))
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            loop_ms = 1000.0 * (time.perf_counter() - loop_start)
            arm_delta_max = float(np.max(np.abs(arm_target - current_arm_q)))
            client_infer_ms = _optional_ms(last_inference.client_ms if last_inference is not None else None)
            server_infer_ms = _optional_ms(last_inference.server_ms if last_inference is not None else None)
            model_infer_ms = _optional_ms(last_inference.model_ms if last_inference is not None else None)
            server_prev_total_ms = _optional_ms(last_inference.prev_total_ms if last_inference is not None else None)

            profile_row = {
                "step": step_idx,
                "chunk_index": chunk_index,
                "active_chunk_steps": active_chunk_steps,
                "chunk_boundary": chunk_boundary,
                "extended_chunk": using_extended_chunk,
                "prefetch_submitted": prefetch_submitted,
                "prefetch_ready": prefetch_ready,
                "loop_ms": loop_ms,
                "obs_ms": timing["obs_ms"],
                "gripper_ms": timing["gripper_ms"],
                "build_ms": timing["build_ms"],
                "policy_wait_ms": timing["policy_wait_ms"],
                "client_infer_ms": client_infer_ms,
                "server_infer_ms": server_infer_ms,
                "model_infer_ms": model_infer_ms,
                "server_prev_total_ms": server_prev_total_ms,
                "action_ms": timing["action_ms"],
                "cmd_ms": timing["cmd_ms"],
                "viz_ms": timing["viz_ms"],
                "sleep_ms": timing["sleep_ms"],
                "overrun_ms": timing["overrun_ms"],
                "arm_delta_max": arm_delta_max,
                "policy_action_jump_max": policy_action_jump_max,
                "arm_cmd_jump_max": arm_cmd_jump_max,
                "gripper_cmd_jump_norm": gripper_cmd_jump_norm,
            }
            if profile_writer is not None:
                profile_writer.writerow(profile_row)
                if step_idx % args.profile_every == 0:
                    profile_file.flush()

            if args.profile_timing and (
                step_idx % args.profile_every == 0
                or chunk_boundary
                or using_extended_chunk
                or timing["policy_wait_ms"] > 1.0
                or timing["overrun_ms"] > 1.0
            ):
                LOGGER.info(
                    "timing step=%d chunk=%d/%d boundary=%s extended=%s prefetch=%s/%s "
                    "loop=%.1fms obs=%.1f build=%.1f wait=%.1f infer client/server/model=%.1f/%.1f/%.1f "
                    "cmd=%.1f sleep=%.1f overrun=%.1f action_jump=%.4f cmd_jump=%.4f grip_jump=%.4f",
                    step_idx,
                    chunk_index,
                    active_chunk_steps,
                    chunk_boundary,
                    using_extended_chunk,
                    prefetch_submitted,
                    prefetch_ready,
                    loop_ms,
                    timing["obs_ms"],
                    timing["build_ms"],
                    timing["policy_wait_ms"],
                    client_infer_ms,
                    server_infer_ms,
                    model_infer_ms,
                    timing["cmd_ms"],
                    timing["sleep_ms"],
                    timing["overrun_ms"],
                    policy_action_jump_max,
                    arm_cmd_jump_max,
                    gripper_cmd_jump_norm,
                )

            step_idx += 1
            chunk_index += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if profile_file is not None:
            profile_file.close()
        if image_client is not None:
            image_client.close()


if __name__ == "__main__":
    main()
