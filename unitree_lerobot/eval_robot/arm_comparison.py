"""Offline trajectory and log utilities. No DDS, robot, or Isaac imports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


JOINTS = tuple(f"{side}_{name}" for side in ("left", "right") for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
))
KP = np.array(([80.] * 4 + [40.] * 3) * 2)
KD = np.array(([3.] * 4 + [1.5] * 3) * 2)
URDF = Path(__file__).parent / "assets/g1/g1d_dex1_gravity.urdf"


def positive(value, name, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class Trajectory:
    joint: str = "left_elbow"
    kind: str = "sine"
    amplitude: float = 0.05
    frequency: float = 0.25
    cycles: int = 5
    ramp: float = 2.0
    hold: float = 5.0

    def __post_init__(self):
        if self.joint not in JOINTS or self.kind not in ("hold", "sine", "point"):
            raise ValueError("Unknown joint or trajectory kind")
        positive(abs(self.amplitude), "absolute amplitude")
        for name in ("frequency", "ramp", "hold"):
            positive(getattr(self, name), name)
        if not isinstance(self.cycles, int) or self.cycles < 1:
            raise ValueError("cycles must be a positive integer")

    @property
    def steady_start(self):
        return self.hold + self.ramp

    @property
    def steady_end(self):
        return self.steady_start + (self.cycles / self.frequency if self.kind == "sine" else self.hold)

    @property
    def duration(self):
        return self.hold if self.kind == "hold" else self.steady_end + self.ramp + self.hold

    def sample(self, t):
        """Offset and phase; sine has quintic amplitude ramps and full steady cycles."""
        if t < 0 or t > self.duration:
            return 0., "outside"
        if self.kind == "hold" or t < self.hold:
            return 0., "pre_hold"
        if t < self.steady_start:
            envelope, phase = smoothstep((t - self.hold) / self.ramp), "ramp_in"
        elif t <= self.steady_end:
            envelope, phase = 1., "steady"
        elif t < self.steady_end + self.ramp:
            envelope, phase = 1 - smoothstep((t - self.steady_end) / self.ramp), "ramp_out"
        else:
            return 0., "post_hold"
        wave = np.sin(2 * np.pi * self.frequency * (t - self.steady_start)) if self.kind == "sine" else 1.
        return float(self.amplitude * envelope * wave), phase

    def preview(self, rate=250.):
        positive(rate, "rate")
        t = np.linspace(0, self.duration, int(np.ceil(self.duration * rate)) + 1)
        offset = np.array([self.sample(v)[0] for v in t])
        velocity = np.gradient(offset, t)
        acceleration = np.gradient(velocity, t)
        return t, offset, velocity, acceleration


def smoothstep(s):
    s = np.clip(s, 0., 1.)
    return s**3 * (10 + s * (-15 + 6 * s))


def joint_limits():
    elements = {j.attrib["name"]: j.find("limit") for j in ET.parse(URDF).getroot().findall("joint")}
    return np.array([[float(elements[f"{j}_joint"].attrib[k]) for k in ("lower", "upper")] for j in JOINTS])


def validate_pose(plan, center, margin=0.05):
    center = np.asarray(center, dtype=float)
    positive(margin, "joint limit margin", allow_zero=True)
    if center.shape != (14,) or not np.isfinite(center).all():
        raise ValueError("Starting arm pose must contain 14 finite radians")
    low, high = center.copy(), center.copy()
    j = JOINTS.index(plan.joint)
    if plan.kind == "sine":
        low[j] -= abs(plan.amplitude)
        high[j] += abs(plan.amplitude)
    elif plan.kind == "point":
        low[j] += min(0, plan.amplitude)
        high[j] += max(0, plan.amplitude)
    limits = joint_limits()
    bad = np.flatnonzero((low < limits[:, 0] + margin) | (high > limits[:, 1] - margin))
    if len(bad):
        raise ValueError(f"Trajectory exceeds joint limits/margin: {[JOINTS[i] for i in bad]}")


def clip_target(target, measured, dt=1 / 250, velocity_limit=20.):
    """Same measured-state-relative, global scaling as G1_29_ArmController."""
    delta = np.asarray(target) - np.asarray(measured)
    return measured + delta / max(1., float(np.max(np.abs(delta))) / (dt * velocity_limit))


ARRAY_FIELDS = ("command_t", "command_q", "command_dq", "command_kp", "command_kd", "command_tau",
                "state_t", "state_q", "state_dq", "state_tau", "desired_t", "desired_q")


@dataclass
class Trial:
    command_t: np.ndarray
    command_q: np.ndarray
    command_dq: np.ndarray
    command_kp: np.ndarray
    command_kd: np.ndarray
    command_tau: np.ndarray
    state_t: np.ndarray
    state_q: np.ndarray
    state_dq: np.ndarray
    state_tau: np.ndarray
    desired_t: np.ndarray
    desired_q: np.ndarray
    metadata: dict

    def validate(self):
        for prefix, fields in (("command", ("q", "dq", "kp", "kd", "tau")),
                               ("state", ("q", "dq", "tau")), ("desired", ("q",))):
            t = getattr(self, f"{prefix}_t")
            if t.ndim != 1 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
                raise ValueError(f"{prefix} timestamps must be finite and strictly increasing")
            if prefix != "desired" and len(t) < 2:
                raise ValueError(f"Need at least two {prefix} samples")
            for field in fields:
                a = getattr(self, f"{prefix}_{field}")
                if a.shape != (len(t), 14) or not np.isfinite(a).all():
                    raise ValueError(f"Invalid {prefix}_{field} shape or nonfinite values")
        if (self.command_kp < 0).any() or (self.command_kd < 0).any():
            raise ValueError("Negative PD gains")
        if self.metadata.get("joint_names") != list(JOINTS):
            raise ValueError("Joint names/order mismatch")
        if self.metadata.get("status", "complete") != "complete":
            raise ValueError("Trial was interrupted/aborted; inspect events.jsonl before using it")
        return self

    def save(self, path):
        self.validate()
        with Path(path).open("xb") as f:
            np.savez_compressed(f, **{name: getattr(self, name) for name in ARRAY_FIELDS},
                                metadata=json.dumps(self.metadata, allow_nan=False))


def load_trial(path, topic=None):
    path = Path(path)
    if path.is_dir():
        path = path / "trial.npz"
    if path.suffix == ".jsonl":
        return load_events(path, topic)
    with np.load(path, allow_pickle=False) as a:
        result = Trial(**{name: a[name] for name in ARRAY_FIELDS}, metadata=json.loads(str(a["metadata"])))
    return result.validate()


def load_events(path, topic=None):
    """Read this runner or the existing passive recorder's JSONL format."""
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta = next((e.copy() for e in events if e.get("stream") == "metadata"), {})
    end = next((e for e in reversed(events) if e.get("stream") == "end"), {})
    if meta and end.get("status") != "complete":
        raise ValueError("No successful end marker; recording is incomplete")
    commands = [e for e in events if e.get("stream") == "command"]
    topics = {e["topic"] for e in commands}
    topic = topic or meta.get("command_topic")
    if topic is None:
        if len(topics) != 1:
            raise ValueError("Select --command-topic: log has zero or multiple command topics")
        topic = next(iter(topics))
    if topics - {topic}:
        raise ValueError("Competing command topics in log; cannot identify the input applied to the robot")
    commands = sorted((e for e in commands if e["topic"] == topic), key=lambda e: e["time_monotonic_ns"])
    states = sorted((e for e in events if e.get("stream") == "state"), key=lambda e: e["time_monotonic_ns"])
    if not commands or not states:
        raise ValueError("Recording needs commands and measured states")
    if any(e.get("crc_valid") is False for e in commands + states):
        raise ValueError("Recording contains invalid CRCs")
    for e in commands:
        if e.get("mode_pr", 0) != 0 or any(e["joints"][j]["mode"] != 1 for j in JOINTS):
            raise ValueError("Replay requires PR mode 0 and all 14 arm motors active")
        if topic == "rt/arm_sdk" and e.get("arm_sdk_weight") != 1.:
            raise ValueError("Replay requires arm_sdk weight 1; partial blending is not identified")
    unique_states, duplicates = [], 0
    for e in states:
        if unique_states and e.get("tick") is not None and e["tick"] == unique_states[-1].get("tick"):
            duplicates += 1
            continue
        unique_states.append(e)
    states = unique_states
    origin = commands[0]["time_monotonic_ns"]
    desired = sorted((e for e in events if e.get("stream") == "desired"), key=lambda e: e["time_monotonic_ns"])
    def times(rows):
        return np.array([(e["time_monotonic_ns"] - origin) / 1e9 for e in rows])
    def values(rows, field):
        return np.array([[e["joints"][j][field] for j in JOINTS] for e in rows], dtype=float).reshape(-1, 14)
    data = {"command_t": times(commands), "state_t": times(states), "desired_t": times(desired),
            "desired_q": np.array([e["q"] for e in desired], dtype=float).reshape(-1, 14)}
    data.update({f"command_{f}": values(commands, f) for f in ("q", "dq", "kp", "kd", "tau")})
    data.update({f"state_{f}": values(states, f if f != "tau" else "tau_est") for f in ("q", "dq", "tau")})
    meta.update(joint_names=list(JOINTS), command_topic=topic, status="complete",
                duplicate_state_ticks=duplicates, source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                time_origin_monotonic_ns=origin)
    if "trajectory_origin_ns" in meta:
        meta["trajectory_offset_s"] = (meta["trajectory_origin_ns"] - origin) / 1e9
    return Trial(**data, metadata=meta).validate()


def previous_indices(t, sample_t):
    """Zero-order hold; -1 before first sample (never borrow a future command)."""
    return np.searchsorted(sample_t, t, side="right") - 1


def interpolate(t, sample_t, values):
    return np.column_stack([np.interp(t, sample_t, values[:, j]) for j in range(values.shape[1])])


def valid_intervals(t, sample_t, max_gap):
    right = np.searchsorted(sample_t, t, side="left")
    within = (t >= sample_t[0]) & (t <= sample_t[-1])
    right = np.clip(right, 0, len(sample_t) - 1)
    left = np.maximum(0, right - 1)
    exact = np.isclose(t, sample_t[right], rtol=0, atol=1e-10)
    return within & (exact | ((sample_t[right] - sample_t[left]) <= max_gap))


def compare(real, sim, rate=100., max_gap=0.1):
    positive(rate, "analysis rate")
    positive(max_gap, "maximum interpolation gap")
    real.validate()
    sim.validate()
    start = max(0., real.state_t[0], sim.state_t[0], real.command_t[0], sim.command_t[0])
    end = min(real.state_t[-1], sim.state_t[-1], real.command_t[-1], sim.command_t[-1])
    t_all = np.arange(start, end, 1 / rate)
    mask = valid_intervals(t_all, real.state_t, max_gap) & valid_intervals(t_all, sim.state_t, max_gap)
    for trial in (real, sim):
        idx = previous_indices(t_all, trial.command_t)
        mask &= (idx >= 0) & ((t_all - trial.command_t[np.maximum(idx, 0)]) <= max_gap)
    t = t_all[mask]
    if len(t) < 3:
        raise ValueError("Insufficient overlapping data after excluding gaps")
    rq, sq = (interpolate(t, run.state_t, run.state_q) for run in (real, sim))
    rdq, sdq = (interpolate(t, run.state_t, run.state_dq) for run in (real, sim))
    ri, si = (previous_indices(t, run.command_t) for run in (real, sim))
    rc, sc = real.command_q[ri], sim.command_q[si]
    diff = sq - rq
    rows = []
    for j, name in enumerate(JOINTS):
        rmse = float(np.sqrt(np.mean(diff[:, j] ** 2)))
        rows.append(dict(joint=name, gap_rmse_rad=rmse, gap_rmse_deg=float(np.rad2deg(rmse)),
                         bias_rad=float(np.mean(diff[:, j])),
                         p95_abs_rad=float(np.percentile(abs(diff[:, j]), 95)),
                         max_abs_rad=float(np.max(abs(diff[:, j]))),
                         velocity_rmse_rad_s=float(np.sqrt(np.mean((sdq[:, j] - rdq[:, j]) ** 2))),
                         real_packet_tracking_rmse_rad=float(np.sqrt(np.mean((rq[:, j] - rc[:, j]) ** 2))),
                         sim_packet_tracking_rmse_rad=float(np.sqrt(np.mean((sq[:, j] - sc[:, j]) ** 2)))))
    command_difference = {field: float(np.max(abs(getattr(real, f"command_{field}")[ri] -
                                                       getattr(sim, f"command_{field}")[si])))
                          for field in ("q", "dq", "kp", "kd", "tau")}
    summary = dict(samples=len(t), excluded_samples=int((~mask).sum()), start_s=float(t[0]), end_s=float(t[-1]),
                   time_alignment="recorded command origin; no fitted shift or time warping",
                   max_command_difference=command_difference, joints=rows)
    plan = real.metadata.get("trajectory")
    if plan and plan["kind"] == "sine":
        trajectory = Trajectory(**plan)
        offset = real.metadata.get("trajectory_offset_s", 0.)
        selected = (t >= trajectory.steady_start + offset) & (t <= trajectory.steady_end + offset)
        if selected.sum() >= 10 and np.ptp(t[selected]) >= 2 / trajectory.frequency:
            w = 2 * np.pi * trajectory.frequency
            x = np.column_stack((np.ones(selected.sum()), np.sin(w * t[selected]), np.cos(w * t[selected])))
            j = JOINTS.index(trajectory.joint)
            cr, cs = (np.linalg.lstsq(x, q[selected, j], rcond=None)[0] for q in (rq, sq))
            ar, ass = float(np.hypot(cr[1], cr[2])), float(np.hypot(cs[1], cs[2]))
            phase = float(np.angle(np.exp(1j * (np.arctan2(cs[2], cs[1]) - np.arctan2(cr[2], cr[1])))))
            summary["sine"] = dict(joint=trajectory.joint, frequency_hz=trajectory.frequency,
                                   real_amplitude_rad=ar, sim_amplitude_rad=ass,
                                   sim_over_real_amplitude=ass / ar if ar > 1e-6 else None,
                                   sim_phase_minus_real_rad=phase if min(ar, ass) > 1e-6 else None,
                                   sim_lag_relative_to_real_ms=-phase / w * 1000 if min(ar, ass) > 1e-6 else None)
    return summary, (t, rq, sq, rc, sc, rdq, sdq)


def write_report(real, sim, output, rate=100., max_gap=0.1):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary, (t, rq, sq, rc, sc, rdq, sdq) = compare(real, sim, rate, max_gap)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary["joints"][0]))
        writer.writeheader()
        writer.writerows(summary["joints"])
    # Leave visible breaks where samples were excluded, rather than drawing
    # straight lines across unobserved intervals.
    breaks = np.r_[False, np.diff(t) > 1.5 / rate]
    for values in (rq, sq, rc, sc, rdq, sdq):
        values[breaks] = np.nan
    for j, name in enumerate(JOINTS):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, rc[:, j], "--", label="real motor target", linewidth=1)
        axes[0].plot(t, sc[:, j], ":", label="sim motor target", linewidth=1)
        if len(real.desired_t):
            k = previous_indices(t, real.desired_t)
            valid = k >= 0
            axes[0].plot(t[valid], real.desired_q[k[valid], j], color="gray", label="requested", alpha=.6)
        axes[0].plot(t, rq[:, j], label="real")
        axes[0].plot(t, sq[:, j], label="sim")
        axes[0].set_ylabel("position [rad]")
        axes[0].legend(loc="best")
        axes[1].plot(t, rdq[:, j], label="real")
        axes[1].plot(t, sdq[:, j], label="sim")
        axes[1].set_ylabel("velocity [rad/s]")
        axes[2].plot(t, sq[:, j] - rq[:, j])
        axes[2].set_ylabel("sim - real [rad]")
        axes[2].set_xlabel("time since first real command [s]")
        for ax in axes:
            ax.grid(alpha=.3)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=130)
        plt.close(fig)
    return summary
