"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import numpy as np

from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_lerobot.eval_robot.make_robot import (
    ARM_CONFIG,
    EE_CONFIG,
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig

from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import to_list, to_scalar

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _expected_action_dim(cfg: EvalRealConfig) -> int:
    arm_dof = int(ARM_CONFIG[cfg.arm]["dof"])

    ee_dof = 0
    ee_key = (cfg.ee or "").lower()
    if ee_key:
        if ee_key not in EE_CONFIG:
            raise ValueError(f"Unknown end-effector '{cfg.ee}'. Available: {list(EE_CONFIG.keys())}")
        ee_dof = int(EE_CONFIG[ee_key]["dof"])

    base_type = getattr(cfg, "base_type", "legs")
    mobile_action_dim = 0
    if base_type != "legs":
        mobile_action_dim = 3 if base_type == "mobile_lift" else 1

    torso_yaw_dim = 1 if cfg.replay_torso_yaw else 0

    return arm_dof + (2 * ee_dof) + mobile_action_dim + torso_yaw_dim


def _torso_yaw_action_index(cfg: EvalRealConfig, arm_dof: int, ee_dof: int) -> int:
    base_type = getattr(cfg, "base_type", "legs")
    mobile_action_dim = 0
    if base_type != "legs":
        mobile_action_dim = 3 if base_type == "mobile_lift" else 1
    return arm_dof + (2 * ee_dof if cfg.ee else 0) + mobile_action_dim


def _episode_row_indices(dataset: LeRobotDataset, episode_index: int) -> list[int]:
    rows = []
    for row_idx, value in enumerate(dataset.hf_dataset["episode_index"]):
        ep_idx = int(value.item()) if hasattr(value, "item") else int(value)
        if ep_idx == episode_index:
            rows.append(row_idx)
    return rows


@parser.wrap()
def replay_main(cfg: EvalRealConfig):
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.replay_torso_yaw and cfg.arm != "G1_29":
        raise ValueError("--replay_torso_yaw is currently supported only with --arm=G1_29.")

    image_client = None
    camera_config = None

    if cfg.visualization:
        rerun_logger = RerunLogger()

    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root or None, episodes=[cfg.episodes])
    episode_rows = _episode_row_indices(dataset, cfg.episodes)
    if not episode_rows:
        available = sorted(
            {
                int(value.item()) if hasattr(value, "item") else int(value)
                for value in dataset.hf_dataset["episode_index"]
            }
        )
        raise ValueError(f"Episode {cfg.episodes} not found. Available episodes: {available}")

    # Validate the episode before any DDS publisher is created.
    first_row_idx = episode_rows[0]
    step = dataset.hf_dataset[first_row_idx]
    first_action = _as_numpy(step["action"])
    expected_dim = _expected_action_dim(cfg)
    if first_action.ndim != 1 or first_action.shape[0] < expected_dim:
        raise ValueError(
            f"Dataset action dim is {first_action.shape}, but {cfg.arm} + {cfg.ee or 'no EE'} expects "
            f"at least {expected_dim} values."
        )
    if first_action.shape[0] != expected_dim:
        if not cfg.replay_torso_yaw and first_action.shape[0] == expected_dim + 1:
            raise ValueError(
                f"Dataset action dim is {first_action.shape}, but {cfg.arm} + {cfg.ee or 'no EE'} expects "
                f"{expected_dim} values. If this dataset includes torso yaw, rerun with "
                "--replay_torso_yaw=true."
            )
        if cfg.replay_torso_yaw:
            raise ValueError(
                f"Dataset action dim is {first_action.shape}, but {cfg.arm} + {cfg.ee or 'no EE'} "
                f"+ torso yaw expects exactly {expected_dim} values."
            )
        if cfg.ee:
            raise ValueError(
                f"Dataset action dim is {first_action.shape}, but {cfg.arm} + {cfg.ee} expects "
                f"{expected_dim} values."
            )
        logger_mp.warning(
            f"Dataset action_dim={first_action.shape[0]}; configured no EE, so only the first "
            f"{expected_dim} arm values will be replayed."
        )
    logger_mp.info(
        f"Validated episode {cfg.episodes}: {len(episode_rows)} frames, action_dim={first_action.shape[0]}."
    )

    if not cfg.send_real_robot:
        logger_mp.info("send_real_robot=false; validation complete, no DDS or robot control initialized.")
        return

    if cfg.visualization:
        image_client, camera_config = setup_image_client(cfg)

    domain_id = 1 if getattr(cfg, "sim", False) else 0
    ChannelFactoryInitialize(domain_id, cfg.network_interface or None)
    robot_interface = setup_robot_interface(cfg)

    """The main control and evaluation loop."""
    # Unpack interfaces for convenience
    arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
        robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
    )
    if cfg.replay_torso_yaw and not hasattr(arm_ctrl, "ctrl_waist_yaw"):
        raise ValueError("--replay_torso_yaw requires an arm controller with ctrl_waist_yaw().")
    torso_yaw_action_idx = _torso_yaw_action_index(cfg, arm_dof, ee_dof)

    logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")

    # init pose
    first_state = _as_numpy(step["observation.state"])
    init_left_arm_pose = first_state[:arm_dof]

    try:
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
        if user_input.lower() == "s":
            # The initial robot pose should match the first recorded frame.
            logger_mp.info("Initializing robot to starting pose...")
            tau = arm_ik.solve_tau(init_left_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_left_arm_pose, tau)
            if cfg.replay_torso_yaw:
                init_torso_yaw = float(first_state[torso_yaw_action_idx])
                arm_ctrl.ctrl_waist_yaw(init_torso_yaw)
                logger_mp.info(f"Initializing torso yaw to {init_torso_yaw:.4f} rad.")
            time.sleep(1)
            for idx, row_idx in enumerate(episode_rows):
                loop_start_time = time.perf_counter()

                left_ee_state = right_ee_state = np.array([])
                action_np = _as_numpy(dataset.hf_dataset[row_idx]["action"])

                # exec action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                logger_mp.info(f"arm_action {arm_action}, tau {tau}")

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.replay_torso_yaw:
                    torso_yaw_action = float(action_np[torso_yaw_action_idx])
                    arm_ctrl.ctrl_waist_yaw(torso_yaw_action)
                    logger_mp.info(f"torso_yaw_action {torso_yaw_action:.4f}")

                if cfg.visualization:
                    observation, current_arm_q = process_images_and_observations(
                        image_client,
                        camera_config,
                        arm_ctrl,
                    )
                    state_parts = [current_arm_q, left_ee_state, right_ee_state]
                    if cfg.replay_torso_yaw:
                        state_parts.append(np.array([arm_ctrl.get_current_waist_yaw()]))
                    state = np.concatenate(state_parts)

                    visualization_data(idx, observation, state, action_np, rerun_logger)

                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    finally:
        if image_client is not None:
            image_client.close()


if __name__ == "__main__":
    replay_main()
