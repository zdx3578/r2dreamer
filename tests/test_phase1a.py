import copy
import unittest

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from dreamer import Dreamer


class BoxSpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


class DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


class FakeReplayBuffer:
    def __init__(self, data, initial):
        self._data = data
        self._initial = initial
        self.updated = None

    def sample(self):
        batch_shape = self._data.shape
        index = [
            torch.arange(batch_shape[1]).repeat(batch_shape[0], 1),
            torch.arange(batch_shape[0]).unsqueeze(1).repeat(1, batch_shape[1]),
        ]
        return self._data.clone(), index, (self._initial[0].clone(), self._initial[1].clone())

    def update(self, index, stoch, deter):
        self.updated = (stoch.clone(), deter.clone())


def make_model_config(cnn_keys, mlp_keys):
    return OmegaConf.create(
        {
            "act_entropy": 3e-4,
            "kl_free": 1.0,
            "imag_horizon": 3,
            "horizon": 15,
            "lamb": 0.95,
            "compile": False,
            "log_grads": False,
            "device": "cpu",
            "rep_loss": "r2dreamer",
            "act": "SiLU",
            "norm": True,
            "use_structured_readout": True,
            "use_effect_model": True,
            "use_goal_progress_head": True,
            "use_reachability_head": True,
            "lr": 1e-4,
            "agc": 0.3,
            "pmin": 1e-3,
            "eps": 1e-20,
            "beta1": 0.9,
            "beta2": 0.999,
            "warmup": 0,
            "slow_target_update": 1,
            "slow_target_fraction": 0.02,
            "loss_scales": {
                "barlow": 0.05,
                "rew": 1.0,
                "con": 1.0,
                "dyn": 1.0,
                "rep": 0.1,
                "policy": 1.0,
                "value": 1.0,
                "repval": 0.3,
                "struct_map": 0.2,
                "struct_obj": 0.2,
                "struct_global": 0.2,
                "delta_map": 1.0,
                "delta_obj": 1.0,
                "delta_global": 1.0,
                "event": 0.5,
                "reach": 0.25,
                "goal": 0.25,
            },
            "r2dreamer": {"lambd": 5e-4},
            "phase1a": {"goal_horizon": 3},
            "rssm": {
                "stoch": 4,
                "deter": 64,
                "hidden": 32,
                "discrete": 4,
                "img_layers": 2,
                "obs_layers": 1,
                "dyn_layers": 1,
                "blocks": 4,
                "act": "SiLU",
                "norm": True,
                "unimix_ratio": 0.01,
                "initial": "learned",
                "device": "cpu",
            },
            "encoder": {
                "cnn_keys": cnn_keys,
                "mlp_keys": mlp_keys,
                "cnn": {
                    "act": "SiLU",
                    "norm": True,
                    "kernel_size": 3,
                    "minres": 1,
                    "depth": 8,
                    "mults": [1, 1, 1, 1],
                },
                "mlp": {
                    "shape": None,
                    "layers": 2,
                    "units": 32,
                    "act": "SiLU",
                    "norm": True,
                    "device": "cpu",
                    "outscale": None,
                    "symlog_inputs": True,
                    "name": "mlp_encoder",
                },
            },
            "reward": {
                "shape": [255],
                "layers": 1,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "symexp_twohot", "bin_num": 255},
                "outscale": 0.0,
                "device": "cpu",
                "symlog_inputs": False,
                "name": "reward",
            },
            "cont": {
                "shape": [1],
                "layers": 1,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "binary"},
                "outscale": 1.0,
                "device": "cpu",
                "symlog_inputs": False,
                "name": "cont",
            },
            "actor": {
                "shape": None,
                "layers": 2,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "device": "cpu",
                "dist": {
                    "cont": {"name": "bounded_normal", "min_std": 0.1, "max_std": 1.0},
                    "disc": {"name": "onehot", "unimix_ratio": 0.01},
                    "multi_disc": {"name": "multi_onehot", "unimix_ratio": 0.01},
                },
                "outscale": 0.01,
                "symlog_inputs": False,
                "name": "actor",
            },
            "critic": {
                "shape": [255],
                "layers": 2,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "device": "cpu",
                "dist": {"name": "symexp_twohot", "bin_num": 255},
                "outscale": 0.0,
                "symlog_inputs": False,
                "name": "value",
            },
            "structured_readout": {
                "layers": 2,
                "hidden": 32,
                "map_slots": 6,
                "map_dim": 8,
                "obj_slots": 4,
                "obj_dim": 8,
                "global_dim": 12,
                "rule_dim": 12,
            },
            "effect_model": {"layers": 2, "hidden": 32, "latent_dim": 24},
            "effect_heads": {"layers": 1, "hidden": 24},
            "reachability_head": {"layers": 1, "hidden": 24},
            "goal_progress_head": {"layers": 1, "hidden": 24},
        }
    )


def make_batch(obs_key, obs_tensor, action_dim):
    batch, time = obs_tensor.shape[:2]
    reward = torch.randn(batch, time)
    is_first = torch.zeros(batch, time, dtype=torch.bool)
    is_last = torch.zeros(batch, time, dtype=torch.bool)
    is_terminal = torch.zeros(batch, time, dtype=torch.bool)
    is_last[:, -1] = True
    is_terminal[:, -1] = True
    return TensorDict(
        {
            obs_key: obs_tensor,
            "action": torch.randn(batch, time, action_dim),
            "reward": reward,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        },
        batch_size=(batch, time),
    )


class Phase1ATest(unittest.TestCase):
    def _run_case(self, obs_space, obs_key, obs_tensor, cnn_keys, mlp_keys):
        torch.manual_seed(0)
        config = make_model_config(cnn_keys=cnn_keys, mlp_keys=mlp_keys)
        restored_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        act_space = BoxSpace((3,))
        agent = Dreamer(config, obs_space, act_space)
        initial_state = agent.get_initial_state(obs_tensor.shape[0])
        replay = FakeReplayBuffer(
            make_batch(obs_key, obs_tensor, act_space.shape[0]),
            (initial_state["stoch"], initial_state["deter"]),
        )

        metrics = agent.update(replay)
        self.assertIn("loss/delta_map", metrics)
        self.assertIn("loss/goal", metrics)
        self.assertIn("phase1a/map_std", metrics)
        self.assertEqual(replay.updated[0].shape[:2], obs_tensor.shape[:2])
        self.assertEqual(replay.updated[1].shape[:2], obs_tensor.shape[:2])

        obs_step = {
            obs_key: obs_tensor[:, 0],
            "is_first": torch.zeros(obs_tensor.shape[0], dtype=torch.bool),
            "is_last": torch.zeros(obs_tensor.shape[0], dtype=torch.bool),
            "is_terminal": torch.zeros(obs_tensor.shape[0], dtype=torch.bool),
            "reward": torch.zeros(obs_tensor.shape[0]),
        }
        action, next_state = agent.act(obs_step, agent.get_initial_state(obs_tensor.shape[0]))
        self.assertEqual(action.shape, (obs_tensor.shape[0], act_space.shape[0]))
        self.assertEqual(next_state["deter"].shape[-1], config.rssm.deter)

        state_dict = copy.deepcopy(agent.state_dict())
        restored = Dreamer(restored_config, obs_space, act_space)
        restored.load_state_dict(state_dict)
        for key in (
            "structured_readout.map_head.weight",
            "effect_model.out.weight",
            "goal_progress_head.out.weight",
        ):
            torch.testing.assert_close(state_dict[key], restored.state_dict()[key])

    def test_image_obs_phase1a_update(self):
        obs_space = DictSpace({"image": BoxSpace((16, 16, 3))})
        obs = torch.randint(0, 256, (2, 4, 16, 16, 3), dtype=torch.uint8)
        self._run_case(obs_space, "image", obs, cnn_keys="image", mlp_keys="^$")

    def test_vector_obs_phase1a_update(self):
        obs_space = DictSpace({"state": BoxSpace((6,))})
        obs = torch.randn(2, 4, 6)
        self._run_case(obs_space, "state", obs, cnn_keys="^$", mlp_keys="state")


if __name__ == "__main__":
    unittest.main()
