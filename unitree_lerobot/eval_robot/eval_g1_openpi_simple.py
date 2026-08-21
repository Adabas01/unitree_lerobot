"""Minimal OpenPI Pi0.5 runner for a real Unitree G1 Dex1 setup.

This intentionally does no smoothing, no prefetch, no per-step clipping, and no
profiling. It executes the action rows returned by the VLA in order.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from typing import Any

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from unitree_lerobot.eval_robot.eval_g1_openpi import TASK_PROMPT
from unitree_lerobot.eval_robot.eval_g1_openpi import _as_numpy
from unitree_lerobot.eval_robot.eval_g1_openpi import _build_policy_observation
from unitree_lerobot.eval_robot.eval_g1_openpi import _infer_action_chunk
from unitree_lerobot.eval_robot.eval_g1_openpi import _move_arms_home
from unitree_lerobot.eval_robot.eval_g1_openpi import _parse_bool
from unitree_lerobot.eval_robot.eval_g1_openpi import _policy_action_to_unitree_action
from unitree_lerobot.eval_robot.eval_g1_openpi import _read_gripper_state
from unitree_lerobot.eval_robot.eval_g1_openpi import _write_gripper_targets
from unitree_lerobot.eval_robot.make_robot import process_images_and_observations
from unitree_lerobot.eval_robot.make_robot import setup_image_client
from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

try:
    from openpi_client import websocket_client_policy
except ImportError as exc:
    raise ImportError(
        "openpi-client is required in the robot environment. Install it with:\n"
        "  pip install -e ~/adam_unitree/openpi-g1d/packages/openpi-client"
    ) from exc


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--image-host", default="192.168.123.164")
    parser.add_argument("--network-interface", default="enp3s0")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=300, help="Number of loop steps; use 0 for unlimited.")
    parser.add_argument("--execute-actions", type=_parse_bool, default=False)
    parser.add_argument("--execute-grippers", type=_parse_bool, default=False)
    parser.add_argument("--gripper-scale", type=float, default=5.4)
    parser.add_argument("--home-before-start", type=_parse_bool, default=False)
    parser.add_argument("--home-duration", type=float, default=4.0)
    parser.add_argument("--home-max-step", type=float, default=0.015)
    parser.add_argument("--home-tolerance", type=float, default=0.05)
    parser.add_argument("--home-timeout", type=float, default=10.0)
    parser.add_argument("--home-open-grippers", type=_parse_bool, default=True)
    parser.add_argument("--home-gripper-open", type=float, default=1.0)
    parser.add_argument("--home-gripper-max-step", type=float, default=0.03)
    parser.add_argument("--prompt", default=TASK_PROMPT)
    args = parser.parse_args()

    args.arm = "G1_29"
    args.ee = "dex1"
    args.motion = False
    args.sim = False
    args.dds_timeout = 8.0
    args.policy_layout = "interleaved"
    return args


def build_observation(
    observation: dict[str, Any],
    current_arm_q: np.ndarray,
    ee_shared_mem: dict[str, Any],
    ee_dof: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gripper_state_raw = _read_gripper_state(ee_shared_mem, ee_dof)
    return _build_policy_observation(observation, current_arm_q, gripper_state_raw, args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    if args.frequency <= 0.0:
        raise ValueError("--frequency must be positive.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0; use 0 for unlimited.")
    if args.gripper_scale <= 0.0:
        raise ValueError("--gripper-scale must be positive.")
    if args.home_duration <= 0.0:
        raise ValueError("--home-duration must be positive.")
    if args.home_max_step <= 0.0:
        raise ValueError("--home-max-step must be positive.")
    if args.home_tolerance <= 0.0:
        raise ValueError("--home-tolerance must be positive.")
    if args.home_timeout <= 0.0:
        raise ValueError("--home-timeout must be positive.")
    if not 0.0 <= args.home_gripper_open <= 1.0:
        raise ValueError("--home-gripper-open must be in [0, 1].")
    if args.home_gripper_max_step <= 0.0:
        raise ValueError("--home-gripper-max-step must be positive.")

    LOGGER.info("Connecting to OpenPI policy server at %s:%d", args.policy_host, args.policy_port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)

    image_client = None
    try:
        LOGGER.info("Connecting to teleimager at %s", args.image_host)
        image_client, camera_config = setup_image_client(args)

        ChannelFactoryInitialize(0, args.network_interface)
        robot_interface = setup_robot_interface(args)
        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]
        ee_dof = int(robot_interface["ee_dof"])

        if args.home_before_start:
            _move_arms_home(arm_ctrl, arm_ik, ee_shared_mem, args)

        LOGGER.info(
            "Simple mode: no smoothing, no prefetch, no action limiting. "
            "execute_actions=%s execute_grippers=%s",
            args.execute_actions,
            args.execute_grippers,
        )
        user_input = input("Enter 's' to start simple OpenPI loop: ")
        if user_input.lower() != "s":
            LOGGER.info("Start signal not received; exiting.")
            return

        chunk = None
        chunk_index = 0
        chunk_id = -1
        loop_dt = 1.0 / args.frequency

        steps = itertools.count() if args.max_steps == 0 else range(args.max_steps)
        for step_idx in steps:
            loop_start = time.perf_counter()

            observation, current_arm_q = process_images_and_observations(image_client, camera_config, arm_ctrl)
            current_arm_q = _as_numpy(current_arm_q).astype(np.float64, copy=False)

            if chunk is None or chunk_index >= len(chunk):
                policy_obs = build_observation(observation, current_arm_q, ee_shared_mem, ee_dof, args)
                inference = _infer_action_chunk(policy, policy_obs)
                chunk = inference.chunk
                chunk_index = 0
                chunk_id += 1
                LOGGER.info(
                    "chunk=%d actions=%s infer_ms=%.1f",
                    chunk_id,
                    chunk.shape,
                    inference.client_ms,
                )

            action = _policy_action_to_unitree_action(chunk[chunk_index], args.policy_layout)
            arm_action = action[:14]
            gripper_action_raw = action[14:16] * args.gripper_scale

            if args.execute_actions:
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                if args.execute_grippers:
                    _write_gripper_targets(ee_shared_mem, gripper_action_raw[0], gripper_action_raw[1])

            if step_idx % 10 == 0:
                LOGGER.info(
                    "step=%d chunk=%d action=%d/%d arm_delta_max=%.4f gripper_raw=(%.3f, %.3f)",
                    step_idx,
                    chunk_id,
                    chunk_index,
                    len(chunk),
                    float(np.max(np.abs(arm_action - current_arm_q))),
                    float(gripper_action_raw[0]),
                    float(gripper_action_raw[1]),
                )

            chunk_index += 1
            time.sleep(max(0.0, loop_dt - (time.perf_counter() - loop_start)))
    finally:
        if image_client is not None:
            image_client.close()


if __name__ == "__main__":
    main()
