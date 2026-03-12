import copy
import pathlib
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from arc3_grid_encoder import Arc3GridEncoder
import distributions as dists
from dreamer import Dreamer
from utils.slot_matching import soft_slot_alignment


class BoxSpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


class DiscreteSpace:
    def __init__(self, n):
        self.n = int(n)
        self.discrete = True
        self.shape = (self.n,)


class DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


class FakeReplayBuffer:
    def __init__(self, data, initial):
        self._data = data
        self._initial = initial
        self.updated = None
        self.updated_priority = None

    def sample(self):
        batch_shape = self._data.shape
        index = [
            torch.arange(batch_shape[1]).repeat(batch_shape[0], 1),
            torch.arange(batch_shape[0]).unsqueeze(1).repeat(1, batch_shape[1]),
        ]
        return self._data.clone(), index, (self._initial[0].clone(), self._initial[1].clone())

    def update(self, index, stoch, deter, priority=None):
        self.updated = (stoch.clone(), deter.clone())
        self.updated_priority = None if priority is None else priority.clone()


class FakeFrozenEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, obs):
        batch = next(iter(obs.values())).shape[0]
        return torch.zeros(batch, self.embed_dim, dtype=torch.float32)


class FakeFrozenRSSM(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = int(feat_dim)

    def obs_step(self, prev_stoch, prev_deter, prev_action, embed, is_first):
        return prev_stoch, prev_deter, None

    def get_feat(self, stoch, deter):
        batch = stoch.shape[0]
        return torch.zeros(batch, self.feat_dim, dtype=torch.float32)

    def img_step(self, stoch, deter, action):
        return stoch, deter


class FakeFrozenActor(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, feat):
        logits = self.logits.unsqueeze(0).expand(feat.shape[0], -1)
        return dists.OneHotDist(logits)


def make_model_config(
    cnn_keys,
    mlp_keys,
    arc3_grid_keys="^$",
    use_objectification=False,
    use_phase2=False,
    use_rule_prediction_consumer=False,
):
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
            "use_objectification": use_objectification,
            "use_operator_bank": use_phase2,
            "use_binding_head": use_phase2,
            "use_signature_head": use_phase2,
            "use_rule_update": use_phase2,
            "use_rule_prediction_consumer": use_rule_prediction_consumer,
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
                "obj_stable": 0.25,
                "obj_local": 0.25,
                "obj_rel": 0.25,
                "op_assign": 0.1,
                "op_top1": 0.1,
                "op_proto": 0.1,
                "op_reuse": 0.1,
                "op_entropy": 0.05,
                "bind_ce": 0.1,
                "bind_consistency": 0.1,
                "sig_scope": 0.1,
                "sig_duration": 0.1,
                "sig_impact": 0.1,
                "rule_update": 0.1,
                "memory_read": 0.1,
                "memory_agreement": 0.1,
                "rule_apply": 0.1,
                "two_step_apply": 0.05,
            },
            "r2dreamer": {"lambd": 5e-4},
            "phase1a": {
                "goal_horizon": 3,
                "use_structure_spatial_recon": True,
                "use_structure_change_targets": False,
                "use_local_change_targets": False,
                "use_direct_delta_targets": False,
                "direct_target_blend": 0.25,
                "reach_horizon": 4,
                "event_ema_decay": 0.99,
                "event_threshold_scale": 1.0,
                "event_target_sharpness": 6.0,
                "event_target_floor": 0.05,
                "struct_recon_scale": 0.35,
                "struct_smooth_scale": 0.5,
                "struct_sparse_scale": 0.25,
                "struct_event_scale": 0.25,
                "goal_event_scale": 0.5,
                "goal_struct_scale": 0.5,
                "goal_risk_scale": 0.5,
            },
            "phase2": {
                "m_obj_threshold": 0.1,
                "match_margin_threshold": 0.02,
                "match_gate_mode": "soft",
                "match_gate_floor": 0.25,
                "memory_write_operator_threshold": 0.14,
                "memory_write_binding_threshold": 0.30,
                "memory_write_alignment_threshold": 0.60,
                "memory_write_apply_error_threshold": 0.10,
                "memory_write_delta_threshold": 0.001,
                "memory_retrieve_temperature": 0.5,
                "memory_usage_logit_scale": 0.5,
                "memory_conf_logit_scale": 1.0,
                "memory_signature_logit_scale": 1.0,
                "memory_ema_decay": 0.99,
                "memory_prototype_decay": 0.95,
                "memory_prototype_min_blend": 0.05,
                "memory_agreement_threshold": 0.7,
                "memory_agreement_delta_threshold": 0.001,
                "use_memory_fusion": True,
            },
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
                "arc3_grid_keys": arc3_grid_keys,
                "arc3_grid": {
                    "num_colors": 16,
                    "num_special_tokens": 1,
                    "token_dim": 8,
                    "depth": 8,
                    "mults": [1, 1, 1, 1],
                    "act": "SiLU",
                    "norm": True,
                    "kernel_size": 3,
                },
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
                "query_track_blend": 0.5,
                "query_track_stopgrad": True,
                "query_conf_threshold": 0.15,
                "query_conf_sharpness": 8.0,
            },
            "effect_model": {"layers": 2, "hidden": 32, "latent_dim": 24},
            "effect_heads": {"layers": 1, "hidden": 24},
            "rule_prediction_consumer": {
                "layers": 1,
                "hidden": 24,
                "residual_scale": 0.03,
                "detach_rule_inputs": True,
                "apply_to_map": False,
                "apply_to_obj": False,
                "apply_to_global": True,
                "start_updates": 0,
                "ramp_updates": 1,
                "gate_enable_mode": "none",
                "gate_threshold": 0.0,
            },
            "structure_decoder": {"hidden": 24, "temperature": 1.0},
            "local_effect_decoder": {"layers": 1, "hidden": 24},
            "reachability_head": {"layers": 1, "hidden": 24},
            "goal_progress_head": {"layers": 1, "hidden": 24},
            "objectification": {
                "match_dim": 16,
                "relation_dim": 16,
                "num_motifs": 4,
                "temperature": 0.5,
                "identity_temperature": 0.25,
                "ema_decay": 0.99,
                "sinkhorn_iters": 10,
                "curriculum_updates": 100,
                "obj_stable_early": 0.5,
                "obj_local_early": 0.15,
                "obj_rel_early": 0.1,
                "w_match": 1.0,
                "w_temp": 1.0,
                "w_smooth": 0.5,
                "w_cycle": 0.5,
                "w_contrast": 0.5,
                "w_teacher": 0.5,
                "w_multistep": 0.5,
                "multistep_offset": 2,
                "w_sparse": 1.0,
                "w_conc": 1.0,
                "w_cf": 0.5,
                "w_pair": 1.0,
                "w_motif": 1.0,
                "w_reuse": 0.5,
            },
            "operator_bank": {
                "layers": 1,
                "hidden": 32,
                "num_operators": 4,
                "operator_dim": 16,
                "temperature": 0.5,
            },
            "binding_head": {
                "layers": 1,
                "hidden": 32,
                "num_bindings": 5,
            },
            "signature_head": {
                "layers": 1,
                "hidden": 32,
            },
            "rule_update": {
                "layers": 1,
                "hidden": 32,
            },
        }
    )


def make_batch(obs_tensors, action_dim):
    first_key = next(iter(obs_tensors))
    batch, time = obs_tensors[first_key].shape[:2]
    reward = torch.randn(batch, time)
    is_first = torch.zeros(batch, time, dtype=torch.bool)
    is_last = torch.zeros(batch, time, dtype=torch.bool)
    is_terminal = torch.zeros(batch, time, dtype=torch.bool)
    is_last[:, -1] = True
    is_terminal[:, -1] = True
    data = dict(obs_tensors)
    data.update(
        {
            "action": torch.randn(batch, time, action_dim),
            "reward": reward,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
    )
    return TensorDict(
        data,
        batch_size=(batch, time),
    )


def make_discrete_batch(obs_tensors, action_dim):
    batch = make_batch(obs_tensors, action_dim)
    action_index = torch.randint(action_dim, batch["action"].shape[:2])
    batch["action"] = F.one_hot(action_index, num_classes=action_dim).to(torch.float32)
    return batch


class Phase1ATest(unittest.TestCase):
    def _run_case(
        self,
        obs_space,
        obs_dict,
        cnn_keys,
        mlp_keys,
        arc3_grid_keys="^$",
        action_shape=(3,),
        use_objectification=False,
        use_phase2=False,
        use_rule_prediction_consumer=False,
    ):
        torch.manual_seed(0)
        config = make_model_config(
            cnn_keys=cnn_keys,
            mlp_keys=mlp_keys,
            arc3_grid_keys=arc3_grid_keys,
            use_objectification=use_objectification,
            use_phase2=use_phase2,
            use_rule_prediction_consumer=use_rule_prediction_consumer,
        )
        restored_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        act_space = BoxSpace(action_shape)
        agent = Dreamer(config, obs_space, act_space)
        first_obs = next(iter(obs_dict.values()))
        initial_state = agent.get_initial_state(first_obs.shape[0])
        replay = FakeReplayBuffer(
            make_batch(obs_dict, sum(act_space.shape)),
            (initial_state["stoch"], initial_state["deter"]),
        )

        metrics = agent.update(replay)
        self.assertIn("loss/delta_map", metrics)
        self.assertIn("loss/goal", metrics)
        self.assertIn("phase1a/map_std", metrics)
        self.assertIn("phase1a/slot_carry_confidence", metrics)
        self.assertIn("phase1a/event_rate", metrics)
        self.assertIn("replay/priority_mean", metrics)
        if use_objectification:
            self.assertIn("loss/obj_stable", metrics)
            self.assertIn("phase1b/m_obj", metrics)
            self.assertIn("phase1b/slot_match_random", metrics)
            self.assertIn("phase1b/slot_match_margin", metrics)
            self.assertIn("phase1b/slot_cycle", metrics)
            self.assertIn("phase1b/slot_identity", metrics)
            self.assertIn("phase1b/slot_teacher", metrics)
            self.assertIn("phase1b/slot_multistep", metrics)
            self.assertIn("phase1b/object_interface", metrics)
            self.assertIn("phase1b/obj_stable_scale", metrics)
        if use_phase2:
            self.assertIn("loss/op_assign", metrics)
            self.assertIn("loss/op_top1", metrics)
            self.assertIn("loss/op_entropy", metrics)
            self.assertIn("loss/memory_read", metrics)
            self.assertIn("loss/memory_agreement", metrics)
            self.assertIn("loss/rule_apply", metrics)
            self.assertIn("loss/two_step_apply", metrics)
            self.assertIn("loss/four_step_apply", metrics)
            self.assertIn("phase2/operator_entropy", metrics)
            self.assertIn("phase2/match_gate_scale", metrics)
            self.assertIn("phase2/operator_top1_conf", metrics)
            self.assertIn("phase2/operator_margin", metrics)
            self.assertIn("phase2/binding_top1_conf", metrics)
            self.assertIn("phase2/memory_conf", metrics)
            self.assertIn("phase2/retrieval_peak", metrics)
            self.assertIn("phase2/memory_read_error", metrics)
            self.assertIn("phase2/memory_agreement_error", metrics)
            self.assertIn("phase2/memory_agreement_coverage", metrics)
            self.assertIn("phase2/rule_apply_error", metrics)
            self.assertIn("phase2/two_step_apply_error", metrics)
            self.assertIn("phase2/two_step_memory_conf", metrics)
            self.assertIn("phase2/two_step_retrieval_agreement", metrics)
            self.assertIn("phase2/two_step_fused_delta_rule_abs", metrics)
            self.assertIn("phase2/four_step_apply_error", metrics)
            self.assertIn("phase2/four_step_memory_conf", metrics)
            self.assertIn("phase2/four_step_retrieval_agreement", metrics)
            self.assertIn("phase2/four_step_fused_delta_rule_abs", metrics)
            if "phase2/seven_step_apply_error" in metrics:
                self.assertIn("phase2/seven_step_memory_conf", metrics)
                self.assertIn("phase2/seven_step_retrieval_agreement", metrics)
                self.assertIn("phase2/seven_step_fused_delta_rule_abs", metrics)
            self.assertIn("phase2/memory_write_alignment", metrics)
            self.assertIn("phase2/memory_write_apply_error", metrics)
            self.assertIn("phase2/memory_write_quality_rate", metrics)
            self.assertIn("phase2/rule_memory_write_rate", metrics)
            self.assertIn("phase2/rule_memory_fresh_usage", metrics)
            self.assertIn("phase2/rule_memory_support_mean", metrics)
            if use_rule_prediction_consumer:
                self.assertIn("phase2/rule_consumer_enabled", metrics)
                self.assertIn("phase2/rule_consumer_map_residual_abs", metrics)
                self.assertIn("phase2/rule_consumer_obj_residual_abs", metrics)
                self.assertIn("phase2/rule_consumer_global_residual_abs", metrics)
                self.assertIn("phase2/rule_consumer_map_residual_ratio", metrics)
        self.assertEqual(replay.updated[0].shape[:2], first_obs.shape[:2])
        self.assertEqual(replay.updated[1].shape[:2], first_obs.shape[:2])
        self.assertIsNotNone(replay.updated_priority)

        obs_step = {
            key: value[:, 0] for key, value in obs_dict.items()
        }
        obs_step.update(
            {
                "is_first": torch.zeros(first_obs.shape[0], dtype=torch.bool),
                "is_last": torch.zeros(first_obs.shape[0], dtype=torch.bool),
                "is_terminal": torch.zeros(first_obs.shape[0], dtype=torch.bool),
                "reward": torch.zeros(first_obs.shape[0]),
            }
        )
        action, next_state = agent.act(obs_step, agent.get_initial_state(first_obs.shape[0]))
        self.assertEqual(action.shape, (first_obs.shape[0], sum(act_space.shape)))
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
        self._run_case(obs_space, {"image": obs}, cnn_keys="image", mlp_keys="^$")

    def test_vector_obs_phase1a_update(self):
        obs_space = DictSpace({"state": BoxSpace((6,))})
        obs = torch.randn(2, 4, 6)
        self._run_case(obs_space, {"state": obs}, cnn_keys="^$", mlp_keys="state")

    def test_eval_repeat_calibration_switches_low_confidence_repeat_to_second_best(self):
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        config.actor_eval = OmegaConf.create(
            {
                "repeat_calibration": True,
                "repeat_threshold": 2,
                "min_top1_prob": 0.5,
                "min_margin": 0.15,
            }
        )
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)
        agent._frozen_encoder = FakeFrozenEncoder(agent.embed_size)
        agent._frozen_rssm = FakeFrozenRSSM(agent.rssm.feat_size)
        agent._frozen_actor = FakeFrozenActor([1.2, 1.1, 0.0])

        obs = {
            "state": torch.zeros(2, 6, dtype=torch.float32),
            "is_first": torch.zeros(2, dtype=torch.bool),
        }
        state = agent.get_initial_state(2)
        state["prev_action"] = F.one_hot(torch.tensor([0, 0]), num_classes=3).to(torch.float32)
        state["eval_repeat_count"] = torch.tensor([2, 1], dtype=torch.int32)

        action, next_state = agent.act(obs, state, eval=True)

        torch.testing.assert_close(action[0], torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(action[1], torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(int(next_state["eval_repeat_count"][0]), 1)
        self.assertEqual(int(next_state["eval_repeat_count"][1]), 2)

    def test_eval_raw_mode_bypasses_repeat_calibration(self):
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        config.actor_eval = OmegaConf.create(
            {
                "repeat_calibration": True,
                "repeat_threshold": 2,
                "min_top1_prob": 0.5,
                "min_margin": 0.15,
            }
        )
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)
        agent._frozen_encoder = FakeFrozenEncoder(agent.embed_size)
        agent._frozen_rssm = FakeFrozenRSSM(agent.rssm.feat_size)
        agent._frozen_actor = FakeFrozenActor([1.2, 1.1, 0.0])

        obs = {
            "state": torch.zeros(1, 6, dtype=torch.float32),
            "is_first": torch.zeros(1, dtype=torch.bool),
        }
        state = agent.get_initial_state(1)
        state["prev_action"] = F.one_hot(torch.tensor([0]), num_classes=3).to(torch.float32)
        state["eval_repeat_count"] = torch.tensor([2], dtype=torch.int32)

        action, next_state, info = agent.act(obs, state, eval=True, eval_policy="raw_mode", return_info=True)

        torch.testing.assert_close(action[0], torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(int(next_state["eval_repeat_count"][0]), 3)
        self.assertAlmostEqual(float(info["actor_top1_prob"][0]), 0.453303, places=5)
        self.assertAlmostEqual(float(info["actor_top1_top2_margin"][0]), 0.043137, places=5)

    def test_phase2_soft_match_gate_uses_slot_match_margin_score(self):
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        config.phase2.match_margin_threshold = 0.2
        config.phase2.match_gate_mode = "soft"
        config.phase2.match_gate_floor = 0.25
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        gate, object_gate, match_gate, warmup_gate = agent._phase2_gate(
            {
                "objectness_score": torch.tensor(0.6),
                "slot_match_margin_score": torch.tensor(0.2),
            }
        )
        self.assertAlmostEqual(float(object_gate), 0.5555555, places=5)
        self.assertAlmostEqual(float(match_gate), 0.25, places=6)
        self.assertAlmostEqual(float(gate), float(object_gate * match_gate * warmup_gate), places=6)

        _, _, strong_match_gate, _ = agent._phase2_gate(
            {
                "objectness_score": torch.tensor(0.6),
                "slot_match_margin_score": torch.tensor(0.6),
            }
        )
        self.assertGreater(float(strong_match_gate), float(match_gate))

    def test_rule_prediction_consumer_is_initially_noop(self):
        config = make_model_config(
            cnn_keys="^$",
            mlp_keys="state",
            use_objectification=True,
            use_phase2=True,
            use_rule_prediction_consumer=True,
        )
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        batch_shape = (2, 4)
        effect_out = {
            "delta_map": torch.randn(*batch_shape, agent.structured_readout.map_slots, agent.structured_readout.map_dim),
            "delta_obj": torch.randn(*batch_shape, agent.structured_readout.obj_slots, agent.structured_readout.obj_dim),
            "delta_global": torch.randn(*batch_shape, agent.structured_readout.global_dim),
            "event_logits": torch.randn(*batch_shape, 1),
        }
        structured = {
            "z_eff": torch.randn(*batch_shape, int(config.effect_model.latent_dim)),
            "effect_out": {key: value.clone() for key, value in effect_out.items()},
        }
        artifact = SimpleNamespace(
            delta_rule_fused=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            rho_next_pred=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
        )

        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)

        torch.testing.assert_close(updated["effect_out"]["delta_map"], effect_out["delta_map"])
        torch.testing.assert_close(updated["effect_out"]["delta_obj"], effect_out["delta_obj"])
        torch.testing.assert_close(updated["effect_out"]["delta_global"], effect_out["delta_global"])
        self.assertEqual(float(metrics["phase2/rule_consumer_map_residual_abs"].detach()), 0.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_obj_residual_abs"].detach()), 0.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_global_residual_abs"].detach()), 0.0)

    def test_phase2_rule_prediction_consumer_update(self):
        obs_space = DictSpace({"state": BoxSpace((6,))})
        obs = {"state": torch.randn(2, 4, 6)}
        self._run_case(
            obs_space,
            obs,
            cnn_keys="^$",
            mlp_keys="state",
            action_shape=(3,),
            use_objectification=True,
            use_phase2=True,
            use_rule_prediction_consumer=True,
        )

    def test_rule_prediction_consumer_global_only_keeps_map_obj_fixed(self):
        config = make_model_config(
            cnn_keys="^$",
            mlp_keys="state",
            use_objectification=True,
            use_phase2=True,
            use_rule_prediction_consumer=True,
        )
        config.rule_prediction_consumer.apply_to_map = False
        config.rule_prediction_consumer.apply_to_obj = False
        config.rule_prediction_consumer.apply_to_global = True
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        with torch.no_grad():
            agent.rule_prediction_consumer.delta_map.bias.fill_(1.0)
            agent.rule_prediction_consumer.delta_obj.bias.fill_(1.0)
            agent.rule_prediction_consumer.delta_global.bias.fill_(1.0)

        batch_shape = (2, 4)
        effect_out = {
            "delta_map": torch.randn(*batch_shape, agent.structured_readout.map_slots, agent.structured_readout.map_dim),
            "delta_obj": torch.randn(*batch_shape, agent.structured_readout.obj_slots, agent.structured_readout.obj_dim),
            "delta_global": torch.randn(*batch_shape, agent.structured_readout.global_dim),
            "event_logits": torch.randn(*batch_shape, 1),
        }
        structured = {
            "z_eff": torch.randn(*batch_shape, int(config.effect_model.latent_dim)),
            "effect_out": {key: value.clone() for key, value in effect_out.items()},
        }
        artifact = SimpleNamespace(
            delta_rule_fused=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            rho_next_pred=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            gate=torch.ones(*batch_shape, 1),
        )

        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)

        torch.testing.assert_close(updated["effect_out"]["delta_map"], effect_out["delta_map"])
        torch.testing.assert_close(updated["effect_out"]["delta_obj"], effect_out["delta_obj"])
        self.assertGreater(float(metrics["phase2/rule_consumer_global_residual_abs"].detach()), 0.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_map_residual_abs"].detach()), 0.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_obj_residual_abs"].detach()), 0.0)

    def test_rule_prediction_consumer_schedule_delays_residuals(self):
        config = make_model_config(
            cnn_keys="^$",
            mlp_keys="state",
            use_objectification=True,
            use_phase2=True,
            use_rule_prediction_consumer=True,
        )
        config.rule_prediction_consumer.start_updates = 100
        config.rule_prediction_consumer.ramp_updates = 50
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        with torch.no_grad():
            agent.rule_prediction_consumer.delta_global.bias.fill_(1.0)

        batch_shape = (2, 4)
        effect_out = {
            "delta_map": torch.randn(*batch_shape, agent.structured_readout.map_slots, agent.structured_readout.map_dim),
            "delta_obj": torch.randn(*batch_shape, agent.structured_readout.obj_slots, agent.structured_readout.obj_dim),
            "delta_global": torch.randn(*batch_shape, agent.structured_readout.global_dim),
            "event_logits": torch.randn(*batch_shape, 1),
        }
        structured = {
            "z_eff": torch.randn(*batch_shape, int(config.effect_model.latent_dim)),
            "effect_out": {key: value.clone() for key, value in effect_out.items()},
        }
        artifact = SimpleNamespace(
            delta_rule_fused=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            rho_next_pred=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            gate=torch.ones(*batch_shape, 1),
        )

        agent._model_updates = 50
        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        torch.testing.assert_close(updated["effect_out"]["delta_global"], effect_out["delta_global"])
        self.assertEqual(float(metrics["phase2/rule_consumer_schedule_scale"].detach()), 0.0)

        agent._model_updates = 124
        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        self.assertAlmostEqual(float(metrics["phase2/rule_consumer_schedule_scale"].detach()), 0.5, places=6)
        self.assertGreater(float(metrics["phase2/rule_consumer_global_residual_abs"].detach()), 0.0)

    def test_rule_prediction_consumer_phase2_gate_threshold_blocks_until_ready(self):
        config = make_model_config(
            cnn_keys="^$",
            mlp_keys="state",
            use_objectification=True,
            use_phase2=True,
            use_rule_prediction_consumer=True,
        )
        config.rule_prediction_consumer.gate_enable_mode = "threshold"
        config.rule_prediction_consumer.gate_threshold = 0.2
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        with torch.no_grad():
            agent.rule_prediction_consumer.delta_global.bias.fill_(1.0)

        batch_shape = (2, 4)
        effect_out = {
            "delta_map": torch.randn(*batch_shape, agent.structured_readout.map_slots, agent.structured_readout.map_dim),
            "delta_obj": torch.randn(*batch_shape, agent.structured_readout.obj_slots, agent.structured_readout.obj_dim),
            "delta_global": torch.randn(*batch_shape, agent.structured_readout.global_dim),
            "event_logits": torch.randn(*batch_shape, 1),
        }
        structured = {
            "z_eff": torch.randn(*batch_shape, int(config.effect_model.latent_dim)),
            "effect_out": {key: value.clone() for key, value in effect_out.items()},
        }
        artifact = SimpleNamespace(
            delta_rule_fused=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            rho_next_pred=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            gate=torch.full((*batch_shape, 1), 0.1),
        )

        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        torch.testing.assert_close(updated["effect_out"]["delta_global"], effect_out["delta_global"])
        self.assertEqual(float(metrics["phase2/rule_consumer_enable_gate"].detach()), 0.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_effective_scale"].detach()), 0.0)

        artifact.gate = torch.full((*batch_shape, 1), 0.3)
        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        self.assertEqual(float(metrics["phase2/rule_consumer_enable_gate"].detach()), 1.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_effective_scale"].detach()), 1.0)
        self.assertGreater(float(metrics["phase2/rule_consumer_global_residual_abs"].detach()), 0.0)

    def test_rule_prediction_consumer_sticky_threshold_latches_once_enabled(self):
        config = make_model_config(
            cnn_keys="^$",
            mlp_keys="state",
            use_objectification=True,
            use_phase2=True,
            use_rule_prediction_consumer=True,
        )
        config.rule_prediction_consumer.gate_enable_mode = "sticky_threshold"
        config.rule_prediction_consumer.gate_threshold = 0.2
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        with torch.no_grad():
            agent.rule_prediction_consumer.delta_global.bias.fill_(1.0)

        batch_shape = (2, 4)
        effect_out = {
            "delta_map": torch.randn(*batch_shape, agent.structured_readout.map_slots, agent.structured_readout.map_dim),
            "delta_obj": torch.randn(*batch_shape, agent.structured_readout.obj_slots, agent.structured_readout.obj_dim),
            "delta_global": torch.randn(*batch_shape, agent.structured_readout.global_dim),
            "event_logits": torch.randn(*batch_shape, 1),
        }
        structured = {
            "z_eff": torch.randn(*batch_shape, int(config.effect_model.latent_dim)),
            "effect_out": {key: value.clone() for key, value in effect_out.items()},
        }
        artifact = SimpleNamespace(
            delta_rule_fused=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            rho_next_pred=torch.randn(*batch_shape, agent.structured_readout.rule_dim),
            gate=torch.full((*batch_shape, 1), 0.1),
        )

        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        torch.testing.assert_close(updated["effect_out"]["delta_global"], effect_out["delta_global"])
        self.assertEqual(float(metrics["phase2/rule_consumer_gate_latched"].detach()), 0.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_enable_gate"].detach()), 0.0)

        artifact.gate = torch.full((*batch_shape, 1), 0.3)
        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        self.assertEqual(float(metrics["phase2/rule_consumer_gate_latched"].detach()), 1.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_enable_gate"].detach()), 1.0)
        self.assertGreater(float(metrics["phase2/rule_consumer_global_residual_abs"].detach()), 0.0)

        artifact.gate = torch.full((*batch_shape, 1), 0.0)
        updated, metrics = agent._apply_rule_prediction_consumer(structured, artifact)
        self.assertEqual(float(metrics["phase2/rule_consumer_gate_latched"].detach()), 1.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_enable_gate"].detach()), 1.0)
        self.assertEqual(float(metrics["phase2/rule_consumer_effective_scale"].detach()), 1.0)

    def test_structure_spatial_recon_survives_without_direct_change_targets(self):
        torch.manual_seed(0)
        config = make_model_config(cnn_keys="image", mlp_keys="^$")
        config.use_structure_decoder = True
        config.use_direct_spatial_targets = False
        config.phase1a.use_structure_spatial_recon = True
        config.phase1a.use_structure_change_targets = False
        obs_space = DictSpace({"image": BoxSpace((16, 16, 3))})
        act_space = BoxSpace((3,))
        agent = Dreamer(config, obs_space, act_space)
        obs = torch.randint(0, 256, (2, 4, 16, 16, 3), dtype=torch.uint8)
        initial_state = agent.get_initial_state(obs.shape[0])
        replay = FakeReplayBuffer(
            make_batch({"image": obs}, sum(act_space.shape)),
            (initial_state["stoch"], initial_state["deter"]),
        )

        metrics = agent.update(replay)

        self.assertIn("loss/spatial_recon", metrics)
        self.assertNotIn("loss/region_map", metrics)
        self.assertNotIn("loss/slot_mask", metrics)

    def test_local_change_targets_can_override_legacy_direct_switch(self):
        torch.manual_seed(0)
        config = make_model_config(cnn_keys="image", mlp_keys="^$")
        config.use_local_decoder = True
        config.use_direct_spatial_targets = False
        config.phase1a.use_local_change_targets = True
        obs_space = DictSpace({"image": BoxSpace((16, 16, 3))})
        act_space = BoxSpace((3,))
        agent = Dreamer(config, obs_space, act_space)
        obs = torch.randint(0, 256, (2, 4, 16, 16, 3), dtype=torch.uint8)
        initial_state = agent.get_initial_state(obs.shape[0])
        replay = FakeReplayBuffer(
            make_batch({"image": obs}, sum(act_space.shape)),
            (initial_state["stoch"], initial_state["deter"]),
        )

        metrics = agent.update(replay)

        self.assertIn("loss/local_change", metrics)
        self.assertIn("loss/local_roi", metrics)
        self.assertIn("loss/local_delta", metrics)

    def test_imagination_mode_mix_can_force_greedy_rollout_actions(self):
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        config.actor_imagination = OmegaConf.create(
            {"mode_mix": 1.0, "mode_mix_start_updates": 0, "mode_mix_ramp_updates": 1}
        )
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)
        agent._frozen_rssm = FakeFrozenRSSM(agent.rssm.feat_size)
        agent._frozen_actor = FakeFrozenActor([1.2, 1.1, 0.0])

        stoch, deter = agent.rssm.initial(2)
        _, imag_action = agent._imagine((stoch, deter), imag_horizon=3)

        expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        for step in range(imag_action.shape[1]):
            torch.testing.assert_close(imag_action[0, step], expected)
            torch.testing.assert_close(imag_action[1, step], expected)

    def test_imagination_mode_mix_ratio_stays_zero_before_start_updates(self):
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        config.actor_imagination = OmegaConf.create(
            {"mode_mix": 0.2, "mode_mix_start_updates": 100, "mode_mix_ramp_updates": 50}
        )
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        agent._model_updates = 50
        self.assertEqual(agent._actor_imagination_mode_mix_ratio(), 0.0)

        agent._model_updates = 124
        self.assertAlmostEqual(agent._actor_imagination_mode_mix_ratio(), 0.1, places=6)

        agent._model_updates = 200
        self.assertAlmostEqual(agent._actor_imagination_mode_mix_ratio(), 0.2, places=6)

    def test_actor_entropy_coeff_decays_only_after_start_updates(self):
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        config.actor_entropy_schedule = OmegaConf.create(
            {"decay": True, "start_updates": 100, "ramp_updates": 50, "min_scale": 0.1}
        )
        obs_space = DictSpace({"state": BoxSpace((6,))})
        act_space = DiscreteSpace(3)
        agent = Dreamer(config, obs_space, act_space)

        base = float(config.act_entropy)
        agent._model_updates = 50
        self.assertAlmostEqual(agent._actor_entropy_coeff(), base, places=12)

        agent._model_updates = 124
        self.assertAlmostEqual(agent._actor_entropy_coeff(), base * 0.55, places=12)

        agent._model_updates = 200
        self.assertAlmostEqual(agent._actor_entropy_coeff(), base * 0.1, places=12)

    def test_arc3_grid_phase1a_update(self):
        obs_space = DictSpace(
            {
                "grid": BoxSpace((64, 64, 1)),
                "valid_mask": BoxSpace((64, 64, 1)),
                "state_flags": BoxSpace((4,)),
                "progress": BoxSpace((8,)),
                "action_context": BoxSpace((8,)),
                "action_mask": BoxSpace((8,)),
            }
        )
        grid = torch.randint(0, 16, (2, 4, 64, 64, 1), dtype=torch.int32)
        obs = {
            "grid": grid,
            "valid_mask": torch.randint(0, 2, (2, 4, 64, 64, 1), dtype=torch.int64).to(torch.float32),
            "state_flags": torch.randn(2, 4, 4),
            "progress": torch.randn(2, 4, 8),
            "action_context": torch.randn(2, 4, 8),
            "action_mask": torch.randint(0, 2, (2, 4, 8), dtype=torch.int64).to(torch.float32),
        }
        self._run_case(
            obs_space,
            obs,
            cnn_keys="valid_mask",
            mlp_keys="state_flags|progress|action_context|action_mask",
            arc3_grid_keys="grid",
            action_shape=(8 + 64 + 64,),
        )
        config = make_model_config(cnn_keys="valid_mask", mlp_keys="state_flags|progress|action_context|action_mask", arc3_grid_keys="grid")
        agent = Dreamer(config, obs_space, BoxSpace((8 + 64 + 64,)))
        self.assertIsInstance(agent.encoder.encoders[0], Arc3GridEncoder)

    def test_arc3_objectification_update(self):
        obs_space = DictSpace(
            {
                "grid": BoxSpace((64, 64, 1)),
                "valid_mask": BoxSpace((64, 64, 1)),
                "state_flags": BoxSpace((4,)),
                "progress": BoxSpace((8,)),
                "action_context": BoxSpace((8,)),
                "action_mask": BoxSpace((8,)),
            }
        )
        obs = {
            "grid": torch.randint(0, 16, (2, 4, 64, 64, 1), dtype=torch.int32),
            "valid_mask": torch.randint(0, 2, (2, 4, 64, 64, 1), dtype=torch.int64).to(torch.float32),
            "state_flags": torch.randn(2, 4, 4),
            "progress": torch.randn(2, 4, 8),
            "action_context": torch.randn(2, 4, 8),
            "action_mask": torch.randint(0, 2, (2, 4, 8), dtype=torch.int64).to(torch.float32),
        }
        self._run_case(
            obs_space,
            obs,
            cnn_keys="valid_mask",
            mlp_keys="state_flags|progress|action_context|action_mask",
            arc3_grid_keys="grid",
            action_shape=(8 + 64 + 64,),
            use_objectification=True,
        )

    def test_arc3_phase2_update(self):
        obs_space = DictSpace(
            {
                "grid": BoxSpace((64, 64, 1)),
                "valid_mask": BoxSpace((64, 64, 1)),
                "state_flags": BoxSpace((4,)),
                "progress": BoxSpace((8,)),
                "action_context": BoxSpace((8,)),
                "action_mask": BoxSpace((8,)),
            }
        )
        obs = {
            "grid": torch.randint(0, 16, (2, 4, 64, 64, 1), dtype=torch.int32),
            "valid_mask": torch.randint(0, 2, (2, 4, 64, 64, 1), dtype=torch.int64).to(torch.float32),
            "state_flags": torch.randn(2, 4, 4),
            "progress": torch.randn(2, 4, 8),
            "action_context": torch.randn(2, 4, 8),
            "action_mask": torch.randint(0, 2, (2, 4, 8), dtype=torch.int64).to(torch.float32),
        }
        self._run_case(
            obs_space,
            obs,
            cnn_keys="valid_mask",
            mlp_keys="state_flags|progress|action_context|action_mask",
            arc3_grid_keys="grid",
            action_shape=(8 + 64 + 64,),
            use_objectification=True,
            use_phase2=True,
        )

    def test_phase1a_arc3_exp_config_merges_to_root(self):
        config_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="configs", overrides=["env=arc3_grid", "+exp=phase1a_arc3"])
        self.assertTrue(bool(cfg.model.use_structured_readout))
        self.assertTrue(bool(cfg.model.use_effect_model))
        self.assertTrue(bool(cfg.model.use_goal_progress_head))
        self.assertTrue(bool(cfg.model.use_reachability_head))
        self.assertEqual(str(cfg.model.encoder.arc3_grid_keys), "grid")

    def test_phase1b_arc3_exp_config_merges_to_root(self):
        config_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="configs", overrides=["env=arc3_grid", "+exp=phase1b_arc3"])
        self.assertTrue(bool(cfg.model.use_objectification))
        self.assertEqual(str(cfg.model.encoder.arc3_grid_keys), "grid")

    def test_phase2_arc3_exp_config_merges_to_root(self):
        config_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="configs", overrides=["env=arc3_grid", "+exp=phase2_arc3"])
        self.assertTrue(bool(cfg.model.use_operator_bank))
        self.assertTrue(bool(cfg.model.use_binding_head))
        self.assertTrue(bool(cfg.model.use_signature_head))
        self.assertTrue(bool(cfg.model.use_rule_update))

    def test_discrete_actor_diagnostics_logged(self):
        torch.manual_seed(0)
        config = make_model_config(cnn_keys="^$", mlp_keys="state")
        obs_space = DictSpace({"state": BoxSpace((6,))})
        obs = {"state": torch.randn(2, 4, 6)}
        act_space = DiscreteSpace(4)
        agent = Dreamer(config, obs_space, act_space)
        initial_state = agent.get_initial_state(obs["state"].shape[0])
        replay = FakeReplayBuffer(
            make_discrete_batch(obs, act_space.n),
            (initial_state["stoch"], initial_state["deter"]),
        )

        metrics = agent.update(replay)

        self.assertIn("actor_top1_prob", metrics)
        self.assertIn("actor_top1_top2_margin", metrics)
        self.assertIn("actor_margin", metrics)
        self.assertIn("actor_entropy", metrics)
        self.assertIn("actor_mode_repeat_rate", metrics)
        self.assertIn("actor_sample_repeat_rate", metrics)

    def test_sinkhorn_slot_alignment_is_doubly_stochastic(self):
        current = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        nxt = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]], dtype=torch.float32)
        match = soft_slot_alignment(current, nxt, temperature=0.25, sinkhorn_iters=32)
        torch.testing.assert_close(match.sum(dim=-1), torch.ones_like(match.sum(dim=-1)), atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(match.sum(dim=-2), torch.ones_like(match.sum(dim=-2)), atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
