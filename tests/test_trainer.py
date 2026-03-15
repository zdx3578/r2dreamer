import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from trainer import OnlineTrainer


class _FakeReplayBuffer:
    def __init__(self):
        self.transitions = []
        self.loaded_state = None

    def count(self):
        return 0

    def add_transition(self, trans):
        self.transitions.append(trans)

    def state_dict(self):
        return {"transitions": len(self.transitions)}

    def load_state_dict(self, state_dict):
        self.loaded_state = dict(state_dict)


class _FakeLogger:
    def __init__(self):
        self.scalars = []
        self.write_steps = []

    def scalar(self, name, value):
        self.scalars.append((name, float(value)))

    def video(self, name, value):
        pass

    def histogram(self, name, value):
        pass

    def write(self, step, fps=False):
        self.write_steps.append((int(step), bool(fps)))


class _FakeParallelEnv:
    def __init__(self, reward_on_step):
        self.env_num = 1
        self._reward_on_step = float(reward_on_step)

    def step(self, action, done):
        done_flag = bool(done[0].item())
        if done_flag:
            obs = TensorDict(
                {
                    "reward": torch.tensor([[0.0]], dtype=torch.float32),
                    "is_first": torch.tensor([[True]], dtype=torch.bool),
                    "is_last": torch.tensor([[False]], dtype=torch.bool),
                    "is_terminal": torch.tensor([[False]], dtype=torch.bool),
                },
                batch_size=(1,),
                device="cpu",
            )
            next_done = torch.tensor([False], dtype=torch.bool, device="cpu")
        else:
            obs = TensorDict(
                {
                    "reward": torch.tensor([[self._reward_on_step]], dtype=torch.float32),
                    "is_first": torch.tensor([[False]], dtype=torch.bool),
                    "is_last": torch.tensor([[True]], dtype=torch.bool),
                    "is_terminal": torch.tensor([[True]], dtype=torch.bool),
                },
                batch_size=(1,),
                device="cpu",
            )
            next_done = torch.tensor([True], dtype=torch.bool, device="cpu")
        return obs, next_done


class _FakeScheduledParallelEnv:
    def __init__(self, reward_schedule):
        self.env_num = 1
        self._reward_schedule = [float(reward) for reward in reward_schedule]
        self._episode_index = 0

    def step(self, action, done):
        done_flag = bool(done[0].item())
        if done_flag:
            obs = TensorDict(
                {
                    "reward": torch.tensor([[0.0]], dtype=torch.float32),
                    "is_first": torch.tensor([[True]], dtype=torch.bool),
                    "is_last": torch.tensor([[False]], dtype=torch.bool),
                    "is_terminal": torch.tensor([[False]], dtype=torch.bool),
                },
                batch_size=(1,),
                device="cpu",
            )
            next_done = torch.tensor([False], dtype=torch.bool, device="cpu")
        else:
            reward = self._reward_schedule[min(self._episode_index, len(self._reward_schedule) - 1)]
            obs = TensorDict(
                {
                    "reward": torch.tensor([[reward]], dtype=torch.float32),
                    "is_first": torch.tensor([[False]], dtype=torch.bool),
                    "is_last": torch.tensor([[True]], dtype=torch.bool),
                    "is_terminal": torch.tensor([[True]], dtype=torch.bool),
                },
                batch_size=(1,),
                device="cpu",
            )
            next_done = torch.tensor([True], dtype=torch.bool, device="cpu")
            self._episode_index += 1
        return obs, next_done


class _FakeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.eval_flags = []
        self.eval_policies = []
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def eval(self):
        super().eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        return self

    def get_initial_state(self, batch_size):
        return TensorDict(
            {
                "stoch": torch.zeros(batch_size, 1, 1, dtype=torch.float32),
                "deter": torch.zeros(batch_size, 1, dtype=torch.float32),
                "prev_action": torch.zeros(batch_size, 2, dtype=torch.float32),
            },
            batch_size=(batch_size,),
        )

    def act(self, obs, state, eval=False, eval_policy="calibrated_mode", return_info=False):
        self.eval_flags.append(bool(eval))
        self.eval_policies.append(eval_policy if eval else "train")
        action = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        next_state = TensorDict(
            {
                "stoch": state["stoch"].clone(),
                "deter": state["deter"].clone(),
                "prev_action": action.clone(),
            },
            batch_size=state.batch_size,
        )
        if not return_info:
            return action, next_state
        info = {
            "actor_top1_prob": torch.tensor([0.75], dtype=torch.float32),
            "actor_top1_top2_margin": torch.tensor([0.25], dtype=torch.float32),
            "actor_mode_repeat": torch.tensor([False]),
            "actor_repeat_valid": torch.tensor([False]),
            "actor_repeat_streak": torch.tensor([1], dtype=torch.int32),
        }
        return action, next_state, info

    def update(self, replay_buffer):
        raise AssertionError("update() should not be reached in this test")


def _make_config(steps, save_every=0):
    return SimpleNamespace(
        steps=steps,
        pretrain=0,
        eval_every=1,
        save_every=save_every,
        eval_episode_num=1,
        sample_eval_episode_num=0,
        eval_gap_checkpoint_threshold=0.0,
        eval_drop_checkpoint_ratio=0.5,
        eval_drop_checkpoint_sample_ratio=0.75,
        video_pred_log=False,
        params_hist_log=False,
        batch_length=1,
        batch_size=1,
        train_ratio=1,
        action_repeat=1,
        update_log_every=1000,
    )


class TrainerEvalSchedulingTest(unittest.TestCase):
    def test_begin_skips_eval_at_step_zero(self):
        trainer = OnlineTrainer(
            _make_config(steps=1),
            _FakeReplayBuffer(),
            _FakeLogger(),
            None,
            _FakeParallelEnv(reward_on_step=1.0),
            _FakeParallelEnv(reward_on_step=2.0),
        )
        agent = _FakeAgent()

        trainer.begin(agent)

        self.assertNotIn(True, agent.eval_flags)

    def test_begin_still_evals_after_positive_step(self):
        logger = _FakeLogger()
        trainer = OnlineTrainer(
            _make_config(steps=2),
            _FakeReplayBuffer(),
            logger,
            None,
            _FakeParallelEnv(reward_on_step=1.0),
            _FakeParallelEnv(reward_on_step=2.0),
        )
        agent = _FakeAgent()

        trainer.begin(agent)

        self.assertIn(True, agent.eval_flags)
        self.assertIn("episode/eval_score", [name for name, _ in logger.scalars])
        self.assertIn("episode/eval_raw_mode_score", [name for name, _ in logger.scalars])
        self.assertIn("episode/eval_mode_score", [name for name, _ in logger.scalars])
        self.assertIn("episode/eval_calibrated_mode_score", [name for name, _ in logger.scalars])
        self.assertIn("episode/eval_actor_top1_prob", [name for name, _ in logger.scalars])
        self.assertIn((1, False), logger.write_steps)

    def test_load_checkpoint_restores_replay_and_trainer_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            replay = _FakeReplayBuffer()
            replay.transitions.extend([1, 2, 3])
            logger = _FakeLogger()
            trainer = OnlineTrainer(
                _make_config(steps=2, save_every=1),
                replay,
                logger,
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeParallelEnv(reward_on_step=2.0),
            )
            agent = _FakeAgent()
            with torch.no_grad():
                agent.weight.fill_(3.0)
            trainer._prev_eval_score = 12.0
            trainer._prev_probe_mode_score = 7.0
            trainer._resume_update_count = 42
            trainer._should_pretrain._once = False
            trainer._should_log._last = 100
            trainer._should_eval._last = 200
            trainer._should_save._last = 300
            trainer.save_latest(agent, step=123)

            restored_replay = _FakeReplayBuffer()
            restored_trainer = OnlineTrainer(
                _make_config(steps=2, save_every=1),
                restored_replay,
                _FakeLogger(),
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeParallelEnv(reward_on_step=2.0),
            )
            restored_agent = _FakeAgent()
            resumed_step = restored_trainer.load_checkpoint(restored_agent, logdir / "latest.pt")

            self.assertEqual(resumed_step, 123)
            self.assertEqual(restored_trainer._resume_step, 123)
            self.assertEqual(restored_trainer._resume_update_count, 42)
            self.assertEqual(restored_trainer._prev_eval_score, 12.0)
            self.assertEqual(restored_trainer._prev_probe_mode_score, 7.0)
            self.assertFalse(restored_trainer._should_pretrain._once)
            self.assertEqual(restored_trainer._should_log._last, 100)
            self.assertEqual(restored_trainer._should_eval._last, 200)
            self.assertEqual(restored_trainer._should_save._last, 300)
            self.assertEqual(restored_replay.loaded_state, {"transitions": 3})
            self.assertEqual(float(restored_agent.weight.item()), 3.0)

    def test_load_checkpoint_rejects_legacy_agent_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            ckpt = logdir / "legacy.pt"
            torch.save({"step": 10, "agent_state_dict": _FakeAgent().state_dict(), "optims_state_dict": {}}, ckpt)

            trainer = OnlineTrainer(
                _make_config(steps=2),
                _FakeReplayBuffer(),
                _FakeLogger(),
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeParallelEnv(reward_on_step=2.0),
            )
            with self.assertRaisesRegex(ValueError, "does not contain replay buffer state"):
                trainer.load_checkpoint(_FakeAgent(), ckpt)

    def test_begin_logs_sample_probe_metrics_when_enabled(self):
        logger = _FakeLogger()
        config = _make_config(steps=2)
        config.sample_eval_episode_num = 1
        trainer = OnlineTrainer(
            config,
            _FakeReplayBuffer(),
            logger,
            None,
            _FakeParallelEnv(reward_on_step=1.0),
            _FakeParallelEnv(reward_on_step=2.0),
            probe_eval_envs=_FakeParallelEnv(reward_on_step=2.0),
            sample_eval_envs=_FakeParallelEnv(reward_on_step=2.0),
        )
        agent = _FakeAgent()

        trainer.begin(agent)

        scalar_map = dict(logger.scalars)
        self.assertEqual(scalar_map["episode/eval_probe_mode_score"], 2.0)
        self.assertEqual(scalar_map["episode/eval_sample_score"], 2.0)
        self.assertEqual(scalar_map["episode/eval_gap"], 0.0)
        self.assertEqual(scalar_map["episode/eval_gap_abs"], 0.0)
        self.assertEqual(scalar_map["episode/eval_gap_checkpoint_saved"], 0.0)

    def test_begin_saves_eval_gap_snapshot_when_gap_is_anomalous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            logger = _FakeLogger()
            config = _make_config(steps=2)
            config.sample_eval_episode_num = 1
            config.eval_gap_checkpoint_threshold = 3.0
            trainer = OnlineTrainer(
                config,
                _FakeReplayBuffer(),
                logger,
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeParallelEnv(reward_on_step=2.0),
                probe_eval_envs=_FakeParallelEnv(reward_on_step=1.0),
                sample_eval_envs=_FakeParallelEnv(reward_on_step=5.0),
            )
            agent = _FakeAgent()

            trainer.begin(agent)

            checkpoint_path = logdir / "checkpoints" / "eval_alert_step_00000001.pt"
            self.assertTrue(checkpoint_path.exists())
            scalar_map = dict(logger.scalars)
            self.assertEqual(scalar_map["episode/eval_gap"], 4.0)
            self.assertEqual(scalar_map["episode/eval_gap_checkpoint_saved"], 1.0)
            self.assertEqual(scalar_map["episode/eval_gap_triggered"], 1.0)
            self.assertEqual(scalar_map["episode/eval_zero_collapse_triggered"], 0.0)
            self.assertEqual(scalar_map["episode/eval_split_drop_triggered"], 0.0)

    def test_begin_saves_eval_alert_on_zero_collapse_without_sample_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            logger = _FakeLogger()
            trainer = OnlineTrainer(
                _make_config(steps=3),
                _FakeReplayBuffer(),
                logger,
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeScheduledParallelEnv(reward_schedule=[2.0, 0.0]),
            )
            agent = _FakeAgent()

            trainer.begin(agent)

            checkpoint_path = logdir / "checkpoints" / "eval_alert_step_00000002.pt"
            self.assertTrue(checkpoint_path.exists())
            scalar_map = dict(logger.scalars)
            self.assertEqual(scalar_map["episode/eval_gap_checkpoint_saved"], 1.0)
            self.assertEqual(scalar_map["episode/eval_zero_collapse_triggered"], 1.0)

    def test_begin_saves_periodic_snapshots_for_positive_steps_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            trainer = OnlineTrainer(
                _make_config(steps=2, save_every=1),
                _FakeReplayBuffer(),
                _FakeLogger(),
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeParallelEnv(reward_on_step=2.0),
            )
            agent = _FakeAgent()

            trainer.begin(agent)

            checkpoint_dir = logdir / "checkpoints"
            snapshot_path = checkpoint_dir / "step_00000001.pt"
            self.assertTrue(snapshot_path.exists())
            self.assertFalse((checkpoint_dir / "step_00000000.pt").exists())
            snapshot = torch.load(snapshot_path, map_location="cpu")
            self.assertEqual(snapshot["step"], 1)
            self.assertIn("agent_state_dict", snapshot)

    def test_save_latest_writes_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            trainer = OnlineTrainer(
                _make_config(steps=1),
                _FakeReplayBuffer(),
                _FakeLogger(),
                logdir,
                _FakeParallelEnv(reward_on_step=1.0),
                _FakeParallelEnv(reward_on_step=2.0),
            )
            agent = _FakeAgent()

            trainer.save_latest(agent, step=123)

            checkpoint_path = logdir / "latest.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["step"], 123)
            self.assertIn("agent_state_dict", checkpoint)


if __name__ == "__main__":
    unittest.main()
