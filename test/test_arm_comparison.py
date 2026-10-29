"""Offline regression tests: no robot SDK, DDS, or simulator needed."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "unitree_lerobot/eval_robot"
sys.path.insert(0, str(SCRIPTS))
from arm_comparison import (JOINTS, KD, KP, Trajectory, Trial, clip_target, compare,
                            load_events, load_trial, previous_indices, validate_pose)


def synthetic_trial(duration=35., lag=0., bias=0.):
    t = np.arange(0., duration + .002, .004)
    q = np.zeros((len(t), 14))
    measured = q.copy()
    q[:, 3] = .05 * np.sin(2 * np.pi * .25 * t)
    measured[:, 3] = .05 * np.sin(2 * np.pi * .25 * (t - lag)) + bias
    dq = np.zeros_like(q)
    dq[:, 3] = .05 * 2 * np.pi * .25 * np.cos(2 * np.pi * .25 * (t - lag))
    zero = np.zeros_like(q)
    return Trial(command_t=t, command_q=q, command_dq=zero.copy(),
                 command_kp=np.tile(KP, (len(t), 1)), command_kd=np.tile(KD, (len(t), 1)),
                 command_tau=zero.copy(), state_t=t.copy(), state_q=measured, state_dq=dq,
                 state_tau=zero.copy(), desired_t=t.copy(), desired_q=q.copy(),
                 metadata=dict(joint_names=list(JOINTS), trajectory=asdict(Trajectory()), status="complete"))


class TrajectoryTests(unittest.TestCase):
    def test_endpoints_and_smooth_transitions(self):
        for kind in ("point", "sine", "hold"):
            p = Trajectory(kind=kind)
            self.assertEqual(p.sample(0)[0], 0)
            self.assertEqual(p.sample(p.duration)[0], 0)
            for t in (p.hold, p.steady_start, p.steady_end, p.steady_end + p.ramp):
                if t >= p.duration:
                    continue
                eps = 1e-5
                left = (p.sample(t)[0] - p.sample(t - eps)[0]) / eps
                right = (p.sample(t + eps)[0] - p.sample(t)[0]) / eps
                self.assertAlmostEqual(left, right, places=4)

    def test_frequency_and_amplitude(self):
        p = Trajectory(amplitude=.07, frequency=.5)
        self.assertAlmostEqual(p.sample(p.steady_start + .5)[0], .07)
        self.assertAlmostEqual(p.sample(p.steady_start + 1.5)[0], -.07)
        self.assertAlmostEqual(p.steady_end - p.steady_start, 10)

    def test_signed_point_and_limits(self):
        p = Trajectory(kind="point", amplitude=-.05)
        self.assertAlmostEqual(p.sample(p.steady_start)[0], -.05)
        validate_pose(p, np.zeros(14))
        center = np.zeros(14)
        center[3] = -1.0
        with self.assertRaises(ValueError):
            validate_pose(p, center)

    def test_invalid_parameters(self):
        for kwargs in (dict(frequency=0), dict(amplitude=float("nan")), dict(ramp=-1), dict(cycles=0)):
            with self.assertRaises(ValueError):
                Trajectory(**kwargs)

    def test_global_target_limiter(self):
        q = np.zeros(14)
        target = q.copy()
        target[:2] = [.16, -.08]
        np.testing.assert_allclose(clip_target(target, q)[:2], [.08, -.04])
        np.testing.assert_allclose(clip_target(target * .1, q), target * .1)

    def test_preview_does_not_import_robot_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Import hook fails if the supposedly offline path even attempts SDK imports.
            code = f'''import runpy, sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.startswith(("unitree_sdk2py", "pinocchio", "isaaclab")):
            raise RuntimeError("Hardware dependency imported: " + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, {str(SCRIPTS)!r})
sys.argv = ["run_g1_arm_trajectory.py", "--trajectory", "hold", "--hold", "0.1", "--output", {str(Path(tmp) / 'preview')!r}]
runpy.run_path({str(SCRIPTS / 'run_g1_arm_trajectory.py')!r}, run_name="__main__")
'''
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "preview/preview.png").is_file())


class ComparisonTests(unittest.TestCase):
    def test_zero_order_hold_never_selects_future_input(self):
        np.testing.assert_array_equal(previous_indices(np.array([-.1, 0., .003, .004, .009]),
                                                      np.array([0., .004, .008])), [-1, 0, 0, 1, 2])

    def test_known_bias_and_phase(self):
        baseline = synthetic_trial()
        biased = synthetic_trial(bias=.02)
        summary, _ = compare(baseline, biased)
        self.assertAlmostEqual(summary["joints"][3]["gap_rmse_rad"], .02, places=7)
        shifted = synthetic_trial(lag=.08)
        summary, _ = compare(baseline, shifted)
        self.assertAlmostEqual(summary["sine"]["sim_lag_relative_to_real_ms"], 80., places=2)
        self.assertGreater(summary["joints"][3]["gap_rmse_rad"], .001)

    def test_excludes_state_gaps(self):
        real, sim = synthetic_trial(), synthetic_trial()
        keep = (real.state_t < 10) | (real.state_t > 12)
        for name in ("t", "q", "dq", "tau"):
            setattr(real, f"state_{name}", getattr(real, f"state_{name}")[keep])
        summary, values = compare(real, sim)
        self.assertGreater(summary["excluded_samples"], 190)
        self.assertFalse(((values[0] > 10) & (values[0] < 12)).any())

    def test_roundtrip_and_no_overwrite(self):
        real = synthetic_trial(duration=.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial.npz"
            real.save(path)
            loaded = load_trial(path)
            np.testing.assert_array_equal(real.command_tau, loaded.command_tau)
            self.assertEqual(real.metadata, loaded.metadata)
            with self.assertRaises(FileExistsError):
                real.save(path)

    def test_invalid_data_rejected(self):
        for problem in ("time", "order", "nan", "abort"):
            trial = synthetic_trial(duration=.1)
            if problem == "time":
                trial.state_t[1] = trial.state_t[0]
            elif problem == "order":
                trial.metadata["joint_names"] = list(reversed(JOINTS))
            elif problem == "nan":
                trial.command_tau[0, 0] = np.nan
            else:
                trial.metadata["status"] = "aborted"
            with self.assertRaises(ValueError):
                trial.validate()

    def test_jsonl_origin_deduplication_and_competing_topics(self):
        events = []
        for i in range(4):
            state = dict(stream="state", time_monotonic_ns=1000000000 + i * 10000000, tick=i // 2,
                         joints={name: dict(q=0., dq=0., tau_est=0.) for name in JOINTS}, crc_valid=True)
            events.append(state)
            if i % 2 == 0:
                events.append(dict(stream="command", topic="rt/lowcmd", time_monotonic_ns=state["time_monotonic_ns"] + 1000,
                                   joints={name: dict(mode=1, q=0., dq=0., kp=80., kd=3., tau=0.) for name in JOINTS}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events))
            run = load_events(path)
            self.assertEqual(run.command_t[0], 0.)
            self.assertLess(run.state_t[0], 0.)
            self.assertEqual(len(run.state_t), 2)
            self.assertEqual(run.metadata["duplicate_state_ticks"], 2)
            events[-2]["topic"] = "rt/arm_sdk"
            path.write_text("\n".join(json.dumps(e) for e in events))
            with self.assertRaises(ValueError):
                load_events(path)


class RunnerLifecycleTests(unittest.TestCase):
    """Exercise the actual execution loop against fake channels and a fake clock."""
    def run_fake(self, output, stale=False, competing=False):
        import run_g1_arm_trajectory as runner
        callbacks, channels, writes = {}, [], []

        class Clock:
            t = 10.
            def monotonic(self):
                return self.t
            def monotonic_ns(self):
                return int(round(self.t * 1e9))
            def sleep(self, duration):
                self.t += max(duration, .000001)
                if "rt/lowstate" in callbacks and not (stale and len(writes) >= 2):
                    callbacks["rt/lowstate"](state())
        clock = Clock()

        def motor():
            return SimpleNamespace(q=0., dq=0., kp=0., kd=0., tau=0., mode=1)
        def state():
            return SimpleNamespace(tick=int(clock.t * 1000), mode_machine=5, motor_state=[motor() for _ in range(35)])
        def command():
            return SimpleNamespace(motor_cmd=[motor() for _ in range(35)], mode_machine=0, mode_pr=0, crc=1)

        class Subscriber:
            def __init__(self, topic, kind):
                self.topic, self.closed = topic, False
                channels.append(self)
            def Init(self, callback):
                callbacks[self.topic] = callback
                if self.topic == "rt/lowstate":
                    callback(state())
                elif competing:
                    callback(command())
            def Close(self):
                self.closed = True
        class Publisher(Subscriber):
            def Init(self):
                pass
            def Write(self, msg):
                writes.append(clock.t)
                callbacks[self.topic](msg)
                return True

        def state_event(msg):
            return dict(stream="state", time_monotonic_ns=clock.monotonic_ns(), tick=msg.tick, crc_valid=True,
                        joints={name: dict(q=0., dq=0., tau_est=0.) for name in JOINTS})
        def command_event(msg, topic):
            return dict(stream="command", time_monotonic_ns=clock.monotonic_ns(), topic=topic, crc_valid=True,
                        mode_pr=msg.mode_pr, arm_sdk_weight=1.,
                        joints={name: vars(msg.motor_cmd[i + 15]).copy() for i, name in enumerate(JOINTS)})
        modules = {
            "unitree_sdk2py.core.channel": SimpleNamespace(ChannelFactoryInitialize=lambda *a: None,
                ChannelPublisher=Publisher, ChannelSubscriber=Subscriber),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": SimpleNamespace(LowCmd_=object, LowState_=object),
            "unitree_sdk2py.idl.default": SimpleNamespace(unitree_hg_msg_dds__LowCmd_=command),
            "unitree_sdk2py.utils.crc": SimpleNamespace(CRC=lambda: SimpleNamespace(Crc=lambda msg: 1)),
            "record_g1_arm_dds": SimpleNamespace(_state_event=state_event, _command_event=command_event),
        }
        args = SimpleNamespace(domain=0, network_interface="fake", command_topic="rt/lowcmd", limit_margin=.05,
                               state_timeout=.02, gravity_feedforward=False, torso_pitch=0., lift_positions=[0., 0.],
                               gripper_position=.0245, target_rate=30., notes="FAKE CHANNEL TEST", output=output,
                               max_torso_drift=.03, max_measured_speed=1.5, max_tracking_error=.25)
        error = None
        with patch.dict(sys.modules, modules), patch.object(runner, "time", clock):
            try:
                runner.execute(args, Trajectory(kind="hold", hold=.08), output)
            except RuntimeError as exc:
                error = exc
        self.assertTrue(all(c.closed for c in channels))
        return error, writes

    def test_completion_saves_a_replayable_trial_and_closes_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            error, writes = self.run_fake(Path(tmp))
            self.assertIsNone(error)
            self.assertGreater(len(writes), 5)
            trial = load_trial(Path(tmp))
            self.assertEqual(trial.command_t[0], 0)
            self.assertEqual(trial.metadata["status"], "complete")

    def test_stale_state_stops_publishing_and_preserves_abort_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            error, writes = self.run_fake(Path(tmp), stale=True)
            self.assertIn("stale", str(error))
            self.assertLess(len(writes), 10)
            result = json.loads((Path(tmp) / "result.json").read_text())
            self.assertEqual(result["status"], "aborted")
            self.assertFalse((Path(tmp) / "trial.npz").exists())
            with self.assertRaises(ValueError):
                load_events(Path(tmp) / "events.jsonl")

    def test_competing_publisher_prevents_any_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            error, writes = self.run_fake(Path(tmp), competing=True)
            self.assertIn("publisher", str(error))
            self.assertEqual(writes, [])


if __name__ == "__main__":
    unittest.main()
