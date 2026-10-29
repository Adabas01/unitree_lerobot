# G1-D Dataset Replay Guide

This guide documents the working G1-D replay setup for a local LeRobot dataset recorded from G1-29 arms with Dex1 grippers, with optional torso yaw.

The tested local dataset is:

```bash
local_dataset/g1_teleop_test
```

The tested robot network interface on the laptop is:

```bash
enx00e04c39c657
```

The tested image server IP is:

```bash
192.168.212.34
```

## Environment

Use the `unitree_lerobot` conda environment:

```bash
cd ~/unitree_g1/unitree_lerobot
conda activate unitree_lerobot
```

Make sure `cyclonedds` is pinned to `0.10.2`. Newer `cyclonedds` versions may initialize but fail to receive `rt/lowstate` from the robot.

```bash
PYTHONNOUSERSITE=1 python -m pip show cyclonedds
```

Expected:

```text
Version: 0.10.2
```

If needed:

```bash
PYTHONNOUSERSITE=1 python -m pip install --force-reinstall cyclonedds==0.10.2
```

Use `PYTHONNOUSERSITE=1` for all commands below. This avoids accidentally importing packages from `~/.local`.

## Check DDS First

Before replaying anything, verify that the laptop can receive `rt/lowstate`:

```bash
cd ~/unitree_g1
conda activate unitree_lerobot

PYTHONNOUSERSITE=1 python experiments/python/g1d_read_state.py enx00e04c39c657
```

Working output looks like:

```text
lowstate: mode_machine=5 imu_rpy=[0.0, 0.0, 0.0]
```

Stop with `Ctrl-C`. Do not use `Ctrl-Z`, because that only suspends the process and can leave DDS participants around.

## Dry Run Dataset Validation

This validates the selected episode and action shape without initializing DDS or robot control:

```bash
cd ~/unitree_g1/unitree_lerobot
conda activate unitree_lerobot

PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id="local_dataset/g1_teleop_test" \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex1" \
  --motion=false \
  --network_interface="enx00e04c39c657" \
  --visualization=false \
  --send_real_robot=false \
  --dds_timeout=8
```

For the current test dataset, expected episode lengths are:

```text
episode 0: 606 frames
episode 1: 413 frames
```

To validate episode 1, change:

```bash
--episodes=1
```

## Replay Arms Only

Use this first. It replays only the 14 arm joints and ignores the two Dex1 gripper action values:

```bash
cd ~/unitree_g1/unitree_lerobot
conda activate unitree_lerobot

PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id="local_dataset/g1_teleop_test" \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="" \
  --motion=false \
  --network_interface="enx00e04c39c657" \
  --visualization=false \
  --send_real_robot=true \
  --dds_timeout=8
```

Before pressing `s`, the arms should hold their current position. Stop immediately if the arms move unexpectedly.

## Replay Arms And Dex1

Use this after arms-only replay works:

```bash
cd ~/unitree_g1/unitree_lerobot
conda activate unitree_lerobot

PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id="local_dataset/g1_teleop_test" \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex1" \
  --motion=false \
  --network_interface="enx00e04c39c657" \
  --visualization=false \
  --send_real_robot=true \
  --dds_timeout=8
```

## Convert Torso Yaw Recordings

If the raw XR recording was collected with `--record-torso-yaw`, convert it with the torso-yaw robot type. This appends `body.torso_yaw` to `observation.state` and `body.torso_yaw_target` to `action`.

Point `--raw-dir` at the task folder you want to convert (for example, `$HOME/datasets/conveyor`). The converter finds episode folders containing `data.json` beneath that directory, including nested recordings. A single episode directory also works. Select a folder containing only torso-yaw recordings when using the torso-yaw robot type.

```bash
cd ~/unitree_g1/unitree_lerobot
conda activate unitree_lerobot

PYTHONNOUSERSITE=1 python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir="$HOME/datasets/g1_torso_yaw_raw" \
  --repo-id="local_dataset/g1_torso_yaw" \
  --robot-type="Unitree_G1_Dex1_TorsoYaw"
```

Raw recordings without `--record-torso-yaw` should keep using `--robot_type="Unitree_G1_Dex1"`.

## Replay Arms, Dex1, And Torso Yaw

Dry-run the converted torso-yaw dataset first:

```bash
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id="local_dataset/g1_torso_yaw" \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex1" \
  --motion=false \
  --network_interface="enx00e04c39c657" \
  --visualization=false \
  --send_real_robot=false \
  --replay_torso_yaw=true \
  --dds_timeout=8
```

Then replay on the robot:

```bash
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id="local_dataset/g1_torso_yaw" \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex1" \
  --motion=false \
  --network_interface="enx00e04c39c657" \
  --visualization=false \
  --send_real_robot=true \
  --replay_torso_yaw=true \
  --dds_timeout=8
```

## Replay With Rerun Visualization

Start the robot image server first, then run:

```bash
cd ~/unitree_g1/unitree_lerobot
conda activate unitree_lerobot

PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id="local_dataset/g1_teleop_test" \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex1" \
  --motion=false \
  --network_interface="enx00e04c39c657" \
  --image_host="192.168.212.34" \
  --visualization=true \
  --send_real_robot=true \
  --dds_timeout=8
```

If camera config fails, check that the image server is reachable at `192.168.212.34:60000`. The client writes a local generated file:

```bash
unitree_lerobot/cam_config_client.yaml
```

That file is ignored by git.

## What Changed In This Branch

The replay path in upstream `unitree_lerobot` was stale. The fixes in this branch are:

- `pyproject.toml`: pin `cyclonedds==0.10.2`.
- `eval_robot/utils/utils.py`: add `image_host`, `network_interface`, and `dds_timeout`.
- `eval_robot/replay_robot.py`: validate the selected episode before DDS, honor `send_real_robot=false`, initialize DDS with the requested interface, use the current image client API, and replay only rows matching `episode_index`.
- `eval_robot/make_robot.py`: pass `dds_timeout` to controllers that support it.
- `eval_robot/robot_control/robot_arm.py`: seed the initial G1-29 arm and waist-yaw targets from the current pose, support commanding waist yaw, and timeout if `rt/lowstate` is not received.
- `utils/constants.py`: add `Unitree_G1_Dex1_TorsoYaw` for recordings that include `body.torso_yaw`.
- `eval_robot/replay_robot.py`: add `--replay_torso_yaw` to validate and replay the extra torso-yaw action.
- `eval_robot/robot_control/robot_hand_unitree.py`: wait for real Dex1 state before starting control, seed gripper targets from current state, and timeout if gripper state is not received.
- `eval_robot/utils/rerun_visualizer.py`: use `rr.Scalars` for `rerun-sdk` 0.26.x.

## Troubleshooting

If replay waits for DDS:

```bash
PYTHONNOUSERSITE=1 python experiments/python/g1d_read_state.py enx00e04c39c657
```

If this does not show `lowstate`, replay will not work. Check the interface name and the `cyclonedds` version.

If Dex1 replay waits for gripper state, check that the Dex1 service is running on the robot and started with the robot-side interface that has a `192.168.123.*` address.

If the Dex1 service segfaults with `ddsi_xt_type_init_impl with invalid type object`, debug the robot-side Dex1 service separately. Arms-only replay does not depend on Dex1.

## Save To Git

Check the changes:

```bash
git status --short
git diff --stat
```

Commit the replay fix and this guide:

```bash
git add .gitignore pyproject.toml README.md docs/g1d_replay.md \
  unitree_lerobot/eval_robot/make_robot.py \
  unitree_lerobot/eval_robot/replay_robot.py \
  unitree_lerobot/eval_robot/robot_control/robot_arm.py \
  unitree_lerobot/eval_robot/robot_control/robot_hand_unitree.py \
  unitree_lerobot/eval_robot/utils/rerun_visualizer.py \
  unitree_lerobot/eval_robot/utils/utils.py

git commit -m "Fix G1-D dataset replay"
```

Push to your fork:

```bash
git push -u origin fix/replay-41c2805
```
