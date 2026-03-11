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

    def count(self):
        return 0

    def add_transition(self, trans):
        self.transitions.append(trans)


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


class _FakeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.eval_flags = []
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

    def act(self, obs, state, eval=False):
        self.eval_flags.append(bool(eval))
        action = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        next_state = TensorDict(
            {
                "stoch": state["stoch"].clone(),
                "deter": state["deter"].clone(),
                "prev_action": action.clone(),
            },
            batch_size=state.batch_size,
        )
        return action, next_state

    def update(self, replay_buffer):
        raise AssertionError("update() should not be reached in this test")


def _make_config(steps, save_every=0):
    return SimpleNamespace(
        steps=steps,
        pretrain=0,
        eval_every=1,
        save_every=save_every,
        eval_episode_num=1,
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
        self.assertIn((1, False), logger.write_steps)

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
