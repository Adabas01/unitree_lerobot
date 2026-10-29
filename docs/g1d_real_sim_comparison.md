This protocol compares G1-D arm motion with the base, lift, and torso held at
matched fixed poses. It measures the control/dynamics gap before introducing
vision, policy inference, moving grippers, or object contacts.

**Run the tools**

The experiment tools are implemented. Use the `unitree_lerobot` environment for
preview, physical recording, and analysis, and `unitree_sim_env` for replay.
Every output directory/file must be new; existing results are never overwritten.

First generate a preview. This command does not import DDS or connect to a robot:

```bash
cd /home/humanoid/adam_unitree/unitree_lerobot
conda activate unitree_lerobot
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/run_g1_arm_trajectory.py \
  --joint left_elbow --trajectory sine --amplitude 0.05 \
  --sine-frequency 0.25 --cycles 5 \
  --output logs/arm_comparison/elbow_preview
```

Inspect `preview.png` and `preview.csv`. The default experiment lasts 34 seconds:
5 seconds initial hold, 2 seconds ramp in, 20 seconds steady sine, 2 seconds
ramp out, and 5 seconds final hold. Offsets are relative to the measured starting
pose; the tool never homes the arms. `--trajectory point` makes a smooth move,
hold, and return; `--trajectory hold` holds for `--hold` seconds.

To record a physical trial, stop other arm command publishers, put the robot in
its established operating mode with the torso/base parked, check the path's
clearance, and retain the normal physical stop control. The runner enforces
joint limits, configured speed/tracking bounds, fresh state, and stable torso
yaw; it is not a collision checker. Both grippers must already be held at the
configured opening; this tool does not command them. The fixed-configuration
arguments describe the physical configuration for gravity and replay. Set them
to the actual values rather than relying on defaults if the torso is tilted,
the lift is raised, or the fingers are closed.

This command **publishes motor commands immediately after initialization**:

```bash
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/run_g1_arm_trajectory.py \
  --joint left_elbow --trajectory sine --amplitude 0.05 \
  --sine-frequency 0.25 --cycles 5 \
  --execute --domain 0 --network-interface YOUR_ROBOT_INTERFACE \
  --command-topic rt/lowcmd \
  --torso-pitch 0 --lift-positions 0 0 --gripper-position 0.0245 \
  --notes "Dex1, wrist cameras attached, no payload; repeat 1" \
  --output logs/arm_comparison/elbow_real_01
```

`rt/lowcmd` matches the current G1-D client default; choose `rt/arm_sdk` only
when that is the interface used by your established robot operating mode.
SDK arm weight is 1. The runner uses the existing G1-29 gains and target limiter,
250 Hz nominal packet publication, 30 Hz target/feedforward updates, and the
G1-D Dex1 gravity model. `--no-gravity-feedforward` disables feedforward for a
separately labeled diagnostic. `--domain 1` can exercise a running DDS simulator,
but the offline replay below is preferable for dynamics comparison.

On completion, Ctrl+C, or a detected fault, publication stops and all channels
close. There is no automatic home or background hold process; behavior after
publication stops depends on the robot's existing operating mode/watchdog.
Keep its normal stop/handover procedure available. Interrupted recordings retain
their raw events and an aborted status; replay/analysis reject them.

The run directory contains `trajectory.json`, preview files, `manifest.json`,
`events.jsonl`, `result.json`, and, after successful completion, `trial.npz`.
The manifest records the actual starting arm/waist pose, declared fixed
configuration, software/model hashes, and options. Raw events retain state ticks,
CRC results, command fields, requested targets, and phase labels.

Replay that recording with gravity and self-collision enabled, using physical
joint constraints to fix every non-arm joint. The replay opens no DDS channels:

```bash
source /home/humanoid/adam_unitree/setup_unitree_sim/activate.sh
cd /home/humanoid/adam_unitree/unitree_sim_isaaclab
PYTHONDONTWRITEBYTECODE=1 python tools/replay_g1_arm_log.py \
  --input ../unitree_lerobot/logs/arm_comparison/elbow_real_01/trial.npz \
  --output ../unitree_lerobot/logs/arm_comparison/elbow_sim_01.npz \
  --headless --device cpu --physics-dt 0.002
```

There is no wall-clock pacing or live watchdog in this replay. Each step applies
the latest timestamped command, including gains, velocity target, and feedforward.
Its metadata records timestamp quantization, skipped commands, and constraint
drift. Use a smaller timestep if commands were skipped. Run again at
`--physics-dt 0.001` with a new output filename to check numerical convergence on
your actual dataset. `--controller legacy` provides a position-only baseline
with the legacy actuator settings, still at the selected offline timestep.
It does not recreate the sorting application's 30 Hz DDS command consumption.
Optional `--armature` and `--friction` apply a shared scalar override to all arm
joints for manual parameter experiments; automatic parameter fitting is not
part of this initial comparison tool.

Generate the comparison report:

```bash
conda activate unitree_lerobot
cd /home/humanoid/adam_unitree/unitree_lerobot
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/compare_g1_arm_runs.py \
  --real logs/arm_comparison/elbow_real_01/trial.npz \
  --sim logs/arm_comparison/elbow_sim_01.npz \
  --output logs/arm_comparison/elbow_report_01
```

The report contains a three-panel PNG for each of the 14 joints, `metrics.csv`,
and `metrics.json`. Metrics include position/velocity gap, target tracking,
signed bias, peak and 95th-percentile error, and steady sine amplitude/phase.
The phase-derived lag is relative to the real response, modulo a sine period;
it is not an independent firmware latency measurement. Input-field differences
are reported too. Samples spanning gaps larger than `--max-gap` (default 0.1 s)
are excluded. No fitted time shift or time warping is applied.

The replay and report also accept raw logs from `record_g1_arm_dds.py`, with
`--command-topic` for explicit topic selection. Logs with competing command
topics, invalid CRCs, partial SDK blending, or unsupported motor modes are
rejected. Old passive logs do not contain declared fixed-configuration metadata;
for those, replay uses `--scene` (the current sorting scene by default). Match
that scene's parked configuration to the real recording before interpreting it.

Offline checks (no robot connections):

```bash
cd /home/humanoid/adam_unitree/unitree_lerobot
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s test -p test_arm_comparison.py -v
```

These tests cover known position bias/phase delay, gaps, limits, preview isolation,
log round trips, and the execution loop using fake channels, including stale-state
and competing-publisher failures. A CPU PhysX smoke replay with synthetic input
also completed. These checks validate tooling, not physical robot behavior or
the accuracy of the robot's model.

REALM first reproduced its real controller, then optimized seven joint friction
and seven armature parameters against paired joint trajectories. It used three
trajectory pairs and CMA-ES followed by an annealing search. This protocol
adapts that idea to the G1-D; its proposed excitation settings are experimental
choices, not settings reported by REALM. See
[REALM, Section III, System identification](https://arxiv.org/html/2512.19562v1#S3).

**What to compare**

Record both the input and the measured response. For an actuator-level replay,
the input is the outgoing motor command after host-side processing:

```text
u(t) = {q_des, dq_des, kp, kd, tau_ff, motor modes, command topic, SDK weight}

real robot:       u(t) -> measured q_real(t), dq_real(t)
simulated robot:  u(t) -> measured q_sim(t),  dq_sim(t)
```

Replay the recorded real commands in simulation. Using measured `q_real` as
the simulated position target would define a different experiment and hide
the real controller's tracking errors. Keep the motor PD feedback active in
simulation; only the recorded command sequence is fixed.

Also distinguish this from a second, useful experiment: feeding the same
high-level trajectory into the entire real and simulated client pipelines.
That measures deployment behavior, including their state-dependent target
limiting and gravity compensation. In that experiment the outgoing packets
can differ. Log them and report the input differences alongside the outputs.

**Relevant behavior in this workspace**

| Component | Current behavior | Consequence |
| --- | --- | --- |
| `G1_29_ArmController` | Nominal 250 Hz publication; arm target clipping depends on current measured state | Identical requested trajectories need not produce identical packets |
| Simple OpenPI runner | Nominal 30 Hz action updates; inference can interrupt those updates | Use a deterministic trajectory source for the initial experiment |
| Gravity feedforward | Computed from measured joint positions, then sent in `tau` | Replay recorded `tau` for actuator identification; recompute it only for the full pipeline experiment |
| Aligned simulator bridge | Uses packet `q`, `dq`, `kp`, `kd`, and `tau` with implicit PhysX drives | A starting point for matching the interface, not proof of matching dynamics |
| Legacy simulator bridge | Uses position targets and its own gains; ignores packet feedforward | Useful baseline, but label it as a different controller |
| Sorting environment | 120 Hz physics, decimation 4, command consumption at 30 Hz | Replaying 250 Hz DDS traffic through it discards intermediate commands |
| Simulated low state | Published at about 100 Hz from snapshots updated after environment steps | Repeated packets are not independent 100 Hz measurements |

At the time of inspection, `scene.json` sets `robot.controller_alignment` to
`true`, whereas `eval_g1_openpi_simple.py` defaults `--controller-alignment` to
`false`. When comparing the full aligned pipelines, explicitly pass
`--controller-alignment true` to the simulated client. That option does not
change the real client's behavior. Capture the effective settings per run;
the defaults in the documentation do not necessarily describe a local config.

See the existing
[controller alignment notes](../../unitree_sim_isaaclab/tasks/g1d_tasks/sort_shapes/CONTROLLER_ALIGNMENT.md)
for the bridge's current approximations, including SDK blending and torque
saturation differences.

**Initial experiment**

1. Choose one verified, collision-free arm posture with the actual Dex1 hands
   and wrist cameras attached. Match its joint angles, torso orientation, lift,
   gripper opening, and payload in simulation. Keep the other joints under
   their normal position hold. A fixed chassis alone does not ensure that
   torso joints remain fixed; monitor them too.
2. Hold the pose for 5 seconds, execute one bounded trajectory, and hold for
   another 5 seconds. Start with one elbow, then shoulder pitch, then a wrist
   joint. The starting posture determines which motion is appropriate; a
   numeric amplitude alone does not establish clearance.
3. Repeat each selected trajectory three times on the real robot to estimate
   repeatability. Then extend to the remaining arm joints and both arms.
4. Replay every recorded input in simulation with the corresponding measured
   initial pose and velocity. Preserve the initial hold and gravity. Do not
   reset the simulated joint state to the real state during the replay.
5. Save plots and per-joint metrics before changing any simulated parameters.

Suggested first trajectories, subject to the selected posture and limits:

| Trajectory | Initial choice | What it reveals |
| --- | --- | --- |
| Static holds | Several arm poses, 5 seconds each | Bias, gravity/load mismatch, pose dependence |
| Smooth point-to-point and return | 0.05 rad displacement over 2 seconds, with a hold | Tracking, settling, direction asymmetry |
| Single-joint sinusoid | 0.05 rad amplitude; separate 0.1, 0.25, and 0.5 Hz trials | Amplitude attenuation, phase lag, velocity dependence |
| Coordinated motion | Several joints, then a task-like reach without contact | Coupling and generalization beyond single-joint tests |

These are conservative starting candidates, not verified robot limits. Preview
the complete path, include joint-limit margins, and use the robot's established
stop procedure during execution. Increase amplitude/frequency only after the
initial traces and physical clearance have been checked.

Use smooth starts and stops. For a point-to-point movement over duration `T`:

```text
s = clip(t / T, 0, 1)
q_des(t) = q_start + (q_end - q_start) * (10*s^3 - 15*s^4 + 6*s^5)
```

For a sinusoid, use `q_des = q_center + A * sin(2*pi*f*t)` during the measured
steady segment, with a smooth amplitude envelope for entry and exit. Analyze
complete steady cycles; five cycles take 50 seconds at 0.1 Hz. The steady
segment has maximum speed `2*pi*f*A` and acceleration `(2*pi*f)^2*A`; validate
the envelope's derivatives as well.

Start in joint space. A Cartesian straight line is a useful later validation
path, but IK adds another source of differences. Solve it once with a continuous
joint solution and reuse that joint trajectory on both systems. Keep the
production command convention: the current client sends `dq_des=0`. Supplying
the analytic trajectory derivative in the motor velocity field would change
the controller being evaluated.

**What to record**

Preserve requested joint targets, outgoing motor fields, measured `q`/`dq`,
reported `tau_est`, host monotonic timestamps, device ticks, and explicit
trajectory-start/phase markers. Record all arm joints, including held joints.
Keep run metadata: joint order and units, initial states, trajectory parameters,
controller mode/gains, feedforward setting/model, hardware attachments, gravity
orientation, software revisions/local changes, and simulation timestep.

The existing
[DDS recorder](../unitree_lerobot/eval_robot/record_g1_arm_dds.py)
already captures outgoing commands and measured arm/waist state. It opens
subscribers only. For an initial recording in the robot environment:

```bash
cd /home/humanoid/adam_unitree/unitree_lerobot
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/record_g1_arm_dds.py \
  --domain 0 --network-interface YOUR_ROBOT_INTERFACE \
  --duration 90 --output logs/arm_comparison/real_trial_01.jsonl
```

For the simulator, use `--domain 1`, the interface configured for simulator DDS
(often `lo` on one machine), and a separate output path. These commands only
record; they do not generate or execute a trajectory. Use unique output names
because the recorder overwrites an existing file.

The recorder does not yet save high-level targets, phase markers, a run manifest,
or simulator application timestamps. Its command timestamp is host reception
time, not evidence of when firmware applied the command. Its on-screen tracking
summary uses the latest command across both topics: analyze one explicitly
selected command topic offline and flag competing publishers. Preserve terminal
drop counts; inspect CRC failures, device-tick gaps, and duplicate state ticks.
Reported torque is diagnostic: neither the robot's `tau_est` nor the simulator's
approximate actuator effort is an independent calibrated torque measurement.

**Timing for simulation replay**

The normal sorting application can provide an initial comparison of the entire
deployed pipeline. It cannot yet provide exact timed replay of every 250 Hz
motor packet. In particular, wall-clock DDS replay against a simulator running
slower than real time can look like a dynamics error.

For parameter fitting, use a dedicated offline replay loop that reads the log
and advances on simulation time, without DDS transport. Apply the latest recorded
command whose relative timestamp is at or before each physics step (zero-order
hold), preserve all command fields, and log the applied command and resulting
state at physics rate. Bypass the client's limiter and feedforward recomputation.
Also bypass the live bridge's wall-clock watchdog/reset hold, or replace their
timing with explicit simulation-time behavior; a fast offline replay must not
depend on wall-clock freshness checks.

Choose a physics step small enough to resolve command changes and test timestep
convergence. For example, compare 500 Hz and 1000 Hz offline runs before trusting
a fit; these are candidate integration rates, not claims about motor firmware.
Record command-time quantization. Keep the current 120 Hz/30 Hz deployment path
as a separate validation condition so a fit in the offline loop is not assumed
to transfer unchanged to the sorting application.

Align runs using a known command-start marker and explicit clock mapping.
Deduplicate repeated simulated state ticks. Resample measured states onto a
common time grid over valid overlapping intervals; use zero-order hold for
commands. Do not interpolate across dropouts. Do not independently normalize
duration or shift measured curves to minimize error: delay is part of the gap.
An additional best-lag diagnostic is useful if reported separately. With host
receipt timestamps, fitted latency is an effective end-to-end delay and cannot
be uniquely attributed to the motor firmware.

**Outputs and interpretation**

Plot requested position, the applied motor position target, measured real
position, and measured simulated position on the same time axis. Also plot
velocity and the real/sim residual. Report each joint separately:

```text
gap_RMSE_j = sqrt(mean((q_sim_j - q_real_j)^2))
real_tracking_RMSE_j = sqrt(mean((q_real_j - q_des_j)^2))
sim_tracking_RMSE_j = sqrt(mean((q_sim_j - q_des_j)^2))
```

Report radians and degrees, mean signed gap, 95th-percentile absolute gap,
maximum gap, static bias, and phase/amplitude error for each sinusoidal frequency.
Distinguish high-level desired positions from packet targets in tracking metrics.
For repeats, show the real-run spread as well as the sim/real difference. Do not
treat thousands of temporally correlated samples as independent trials.

Forward kinematics with a common model can translate joint errors into estimated
tool-position/orientation errors. It does not independently validate link
geometry, mounting offsets, or compliance; those need external pose measurement.
Set acceptance thresholds from task clearance and repeatability rather than
assuming that a particular joint RMSE is sufficient for manipulation.

If calibration is needed, check joint mapping, initial pose, gains, feedforward,
command consumption, and delay first. Then fit a small physically constrained
set of friction and armature parameters. Expand to masses, COM offsets, or
actuator saturation only when the data support doing so. A position-only fit
need not identify unique physical parameters.

Split entire trajectories into fitting and validation sets. Reserve different
poses, speeds, and coordinated movements for validation, and report baseline
versus calibrated results on those unseen trajectories. Keep the real controller
fixed while adjusting the simulator. A later full-pipeline test should restore
state-dependent limiting and gravity compensation on each system.

The immediate experimental deliverable is a baseline report for static holds
and one-joint motions. The tools above implement trajectory execution, recording,
simulation-time replay, and reporting. Collecting physical trials and interpreting
their differences remain necessary before claiming a measured real/sim match.
