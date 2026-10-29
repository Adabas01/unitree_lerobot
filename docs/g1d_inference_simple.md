# G1-D Inference: Simple Startup

Use four terminals. Start them in the numbered order below and leave each service running.
This guide assumes the software and Python environments are already set up.

| | Left | Right |
| --- | --- | --- |
| Top | **3. Policy server** — inference laptop | **1. Dex1 service** — SSH to robot |
| Bottom | **4. Policy client** — laptop connected to robot | **2. Tele Imager** — SSH to robot |

The policy server can run on a different laptop. Run its terminal on that laptop (or SSH into it).
The client must be able to reach both the robot and the policy server.

## 1. Top Right: Robot Dex1 Service

Connect to the robot:

```bash
ssh unitree@192.168.123.164
```

If asked:

```text
ros:foxy(1) noetic(2) ?
```

Enter `1` for ROS Foxy.

Start the gripper server:

```bash
cd ~/dex1_1_service/bin
sudo ./dex1_1_gripper_server
```

Good sign:

```text
Dex1-1 Gripper Server started.
```

Leave this running.

## 2. Bottom Right: Robot Tele Imager

Connect in a second terminal:

```bash
ssh unitree@192.168.123.164
```

If asked for ROS, enter `1` again.

Start the image server:

```bash
teleimager-server
```

Leave this running.

## 3. Top Left: Policy Server

On the laptop that will run the policy, open your existing policy-server environment.
The repository and checkpoint paths below are from the screenshot; adjust them if they differ on this laptop.

```bash
cd ~/adam_unitree/ichrpi

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_g1d_sort_lora \
  --policy.dir="$HOME/adam_unitree/openpi-g1d/checkpoints/pi05_g1d_sort_lora/g1d_sort_lora/4999"
```

Wait for the policy to finish loading and the server to start listening on port `8000`.
Leave this running.

## 4. Bottom Left: Policy Client

On the laptop connected to the robot, open your existing policy-client environment.

The command defaults to `localhost` for running the server and client on the same laptop.
If the policy server runs on a separate laptop, replace `localhost` with that laptop's IP address reachable from this laptop.
Also replace `enp3s0` if your robot network interface has a different name.

This command enables robot and gripper movement, with homing before inference starts.

```bash
cd ~/adam_unitree/unitree_lerobot

POLICY_HOST="localhost"

PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/eval_g1_openpi_simple.py \
  --policy-host "$POLICY_HOST" \
  --policy-port 8000 \
  --image-host 192.168.123.164 \
  --network-interface enp3s0 \
  --frequency 30 \
  --max-steps 0 \
  --execute-actions true \
  --execute-grippers true \
  --gripper-scale 5.4 \
  --home-before-start true
```

Keep all four terminals open while inference runs.

## Stop

Press `Ctrl+C` in the policy-client terminal first. Then stop the policy server,
Tele Imager, and Dex1 service with `Ctrl+C` when finished.
