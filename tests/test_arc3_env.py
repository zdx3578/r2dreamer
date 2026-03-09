import sys
import types
import unittest

import numpy as np

from envs.arc3 import (
    Arc3Grid,
    decode_arc_action,
    derive_arc_reward,
    encode_action_mask,
    encode_arc_grid,
    encode_arc_progress,
    encode_arc_state_flags,
    encode_state_flags,
    normalize_arc_frame,
)


class _FakeActionData:
    def __init__(self):
        self._data = {}

    def model_dump(self):
        return dict(self._data)


class _FakeGameAction:
    _id_to_name = {
        0: "RESET",
        1: "ACTION1",
        2: "ACTION2",
        3: "ACTION3",
        4: "ACTION4",
        5: "ACTION5",
        6: "ACTION6",
        7: "ACTION7",
    }

    def __init__(self, action_id):
        self.value = int(action_id)
        self.name = self._id_to_name[int(action_id)]
        self.action_data = _FakeActionData()

    def is_complex(self):
        return self.value == 6

    def set_data(self, data):
        self.action_data._data = dict(data)

    @classmethod
    def from_id(cls, action_id):
        return cls(action_id)


class _FakeRawFrame:
    def __init__(self, frame, state_name, levels_completed, win_levels, available_actions):
        self.frame = frame
        self.state = types.SimpleNamespace(name=state_name)
        self.levels_completed = levels_completed
        self.win_levels = win_levels
        self.available_actions = available_actions


class _FakeArcEnv:
    def __init__(self):
        self.observation_space = _FakeRawFrame(
            frame=[[0 for _ in range(64)] for _ in range(64)],
            state_name="NOT_PLAYED",
            levels_completed=0,
            win_levels=1,
            available_actions=[0],
        )

    def step(self, action, data=None, reasoning=None):
        if action.value == 0:
            frame = [[0 for _ in range(64)] for _ in range(64)]
            frame[4][5] = 3
            return _FakeRawFrame(frame, "PLAYING", 0, 1, [1, 2, 3, 4, 5, 6, 7])
        frame = [[0 for _ in range(64)] for _ in range(64)]
        frame[6][7] = 5
        return _FakeRawFrame(frame, "WIN", 1, 1, [0])


class _FakeArcade:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def make(self, game_id):
        return _FakeArcEnv()


class _FakeOperationMode:
    NORMAL = "normal"
    OFFLINE = "offline"
    ONLINE = "online"

    def __call__(self, value):
        return value


class Arc3EnvTest(unittest.TestCase):
    def test_grid_helpers(self):
        nested = [[[1, 2], [3, 4]]]
        grid = normalize_arc_frame(nested, (4, 4))
        self.assertEqual(grid[0, 0], 1)
        token = encode_arc_grid(grid[:2, :2], 8, "token")
        self.assertEqual(token.shape, (2, 2, 1))
        self.assertEqual(token.dtype, np.int32)
        onehot = encode_arc_grid(grid[:2, :2], 8, "onehot")
        self.assertEqual(onehot.shape, (2, 2, 8))
        scalar = encode_arc_grid(grid[:2, :2], 8, "scalar")
        self.assertEqual(scalar.shape, (2, 2, 1))
        np.testing.assert_array_equal(encode_state_flags("WIN"), np.array([0, 0, 1, 0], dtype=np.float32))
        rich_state = encode_arc_state_flags("WIN", full_reset=True, action_id=6, action_data={"x": 2, "y": 3})
        self.assertEqual(rich_state.shape, (8,))
        rich_progress = encode_arc_progress(
            levels_completed=1,
            win_levels=4,
            available_actions=[1, 2, 6],
            action_count=8,
            action_id=6,
            action_data={"x": 2, "y": 3},
            size=(4, 4),
        )
        self.assertEqual(rich_progress.shape, (8,))
        np.testing.assert_array_equal(
            encode_action_mask([0, 3, 6], 8),
            np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32),
        )

    def test_decode_and_reward(self):
        action = np.zeros(8 + 64 + 64, dtype=np.float32)
        action[6] = 1.0
        action[8 + 11] = 1.0
        action[8 + 64 + 13] = 1.0
        self.assertEqual(decode_arc_action(action, 8, 64, 64), (6, 11, 13))
        reward = derive_arc_reward(0, "PLAYING", 1, "WIN", reward_per_level=1.0, reward_win=10.0, reward_loss=-2.0)
        self.assertEqual(reward, 11.0)

    def test_arc3_reset_and_step(self):
        fake_arc_agi = types.ModuleType("arc_agi")
        fake_arc_agi.Arcade = _FakeArcade
        fake_arc_agi.OperationMode = _FakeOperationMode()
        fake_arcengine = types.ModuleType("arcengine")
        fake_arcengine.GameAction = _FakeGameAction

        old_arc_agi = sys.modules.get("arc_agi")
        old_arcengine = sys.modules.get("arcengine")
        sys.modules["arc_agi"] = fake_arc_agi
        sys.modules["arcengine"] = fake_arcengine
        try:
            env = Arc3Grid("ls20")
            obs = env.reset()
            self.assertEqual(obs["grid"].shape, (64, 64, 1))
            self.assertEqual(obs["grid"].dtype, np.int32)
            self.assertEqual(int(obs["grid"][4, 5, 0]), 3)
            self.assertEqual(obs["state_flags"].shape, (8,))
            self.assertEqual(obs["progress"].shape, (8,))
            self.assertTrue(bool(obs["is_first"]))
            self.assertEqual(int(obs["action_mask"][5]), 1)

            sampled = env.action_space.sample()
            self.assertEqual(sampled.shape, (8 + 64 + 64,))

            action = np.zeros(8 + 64 + 64, dtype=np.float32)
            action[5] = 1.0
            action[8 + 7] = 1.0
            action[8 + 64 + 6] = 1.0
            next_obs, reward, done, info = env.step(action)
            self.assertTrue(done)
            self.assertEqual(reward, 11.0)
            self.assertTrue(bool(next_obs["is_terminal"]))
            self.assertEqual(int(next_obs["grid"][6, 7, 0]), 5)
            self.assertEqual(info["levels_completed"], 1)
            self.assertIn("full_reset", info)
            self.assertIn("action_id", info)
        finally:
            if old_arc_agi is None:
                sys.modules.pop("arc_agi", None)
            else:
                sys.modules["arc_agi"] = old_arc_agi
            if old_arcengine is None:
                sys.modules.pop("arcengine", None)
            else:
                sys.modules["arcengine"] = old_arcengine


if __name__ == "__main__":
    unittest.main()
