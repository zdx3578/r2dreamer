import unittest
from types import SimpleNamespace

import torch

from dreamer import Dreamer
from rule_apply import RuleApply
from rule_memory import RuleMemory


class Phase2RuleExecutionTest(unittest.TestCase):
    def test_rule_memory_update_and_retrieve(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.0, memory_prototype_decay=0.95, memory_retrieve_temperature=1.0),
            num_operators=2,
            num_bindings=3,
            signature_dim=3,
            rule_dim=4,
        )
        q_u = torch.tensor([[[1.0, 0.0]]])
        q_b = torch.tensor([[[0.0, 1.0, 0.0]]])
        q_sigma = torch.tensor([[[0.2, 0.4, 0.6]]])
        delta = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        write_mask = torch.tensor([[[True]]])

        stats = memory.update(q_u, q_b, q_sigma, delta, write_mask)
        out = memory.retrieve(q_u, q_b, q_sigma)

        torch.testing.assert_close(out["memory_delta_rule"], delta)
        torch.testing.assert_close(out["memory_signature_proto"], q_sigma)
        self.assertGreater(float(out["memory_conf"].item()), 0.0)
        self.assertAlmostEqual(float(stats["write_rate"]), 1.0)

    def test_rule_memory_keeps_first_write_magnitude_with_smoothing_enabled(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.99, memory_prototype_decay=0.95, memory_retrieve_temperature=1.0),
            num_operators=2,
            num_bindings=2,
            signature_dim=3,
            rule_dim=3,
        )
        q_u = torch.tensor([[[1.0, 0.0]]])
        q_b = torch.tensor([[[1.0, 0.0]]])
        q_sigma = torch.tensor([[[0.2, 0.4, 0.6]]])
        delta = torch.tensor([[[0.5, 1.5, 2.5]]])

        memory.update(q_u, q_b, q_sigma, delta, torch.tensor([[[True]]]))
        out = memory.retrieve(q_u, q_b, q_sigma)

        torch.testing.assert_close(out["memory_delta_rule"], delta)
        torch.testing.assert_close(out["memory_signature_proto"], q_sigma)

    def test_rule_memory_can_correct_early_wrong_prototype(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.99, memory_prototype_decay=0.5, memory_retrieve_temperature=1.0),
            num_operators=1,
            num_bindings=1,
            signature_dim=2,
            rule_dim=2,
        )
        q_u = torch.tensor([[[1.0]]])
        q_b = torch.tensor([[[1.0]]])
        q_sigma = torch.tensor([[[1.0, 0.0]]])

        memory.update(q_u, q_b, q_sigma, torch.tensor([[[2.0, 0.0]]]), torch.tensor([[[True]]]))
        for _ in range(4):
            memory.update(q_u, q_b, q_sigma, torch.tensor([[[0.0, 4.0]]]), torch.tensor([[[True]]]))

        out = memory.retrieve(q_u, q_b, q_sigma)
        self.assertLess(float(out["memory_delta_rule"][0, 0, 0]), 0.3)
        self.assertGreater(float(out["memory_delta_rule"][0, 0, 1]), 3.7)

    def test_rule_memory_keeps_operator_binding_separated(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.0, memory_prototype_decay=0.95, memory_retrieve_temperature=1.0),
            num_operators=2,
            num_bindings=3,
            signature_dim=3,
            rule_dim=2,
        )
        q_u = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        )
        q_b = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        )
        q_sigma = torch.tensor(
            [
                [[0.1, 0.2, 0.3]],
                [[0.3, 0.2, 0.1]],
            ]
        )
        delta = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 2.0]],
            ]
        )
        write_mask = torch.tensor([[[True]], [[True]]])

        memory.update(q_u, q_b, q_sigma, delta, write_mask)
        first = memory.retrieve(q_u[:1], q_b[:1], q_sigma[:1])
        second = memory.retrieve(q_u[1:], q_b[1:], q_sigma[1:])

        torch.testing.assert_close(first["memory_delta_rule"], delta[:1])
        torch.testing.assert_close(second["memory_delta_rule"], delta[1:])

    def test_rule_memory_uses_signature_to_sharpen_retrieval(self):
        memory = RuleMemory(
            SimpleNamespace(
                memory_ema_decay=0.0,
                memory_prototype_decay=0.95,
                memory_retrieve_temperature=0.1,
                memory_usage_logit_scale=0.0,
                memory_conf_logit_scale=0.0,
                memory_signature_logit_scale=4.0,
            ),
            num_operators=2,
            num_bindings=2,
            signature_dim=3,
            rule_dim=2,
        )
        memory.update(
            torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[0.0, 2.0]],
                ]
            ),
            torch.tensor([[[True]], [[True]]]),
        )

        out = memory.retrieve(
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        )

        self.assertGreater(float(out["memory_delta_rule"][0, 0, 0]), 0.9)
        self.assertLess(float(out["memory_delta_rule"][0, 0, 1]), 0.2)
        self.assertGreater(float(out["memory_top_weight"].item()), 0.9)

    def test_rule_memory_skips_write_when_mask_is_false(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.0, memory_prototype_decay=0.95, memory_retrieve_temperature=1.0),
            num_operators=2,
            num_bindings=2,
            signature_dim=3,
            rule_dim=2,
        )
        stats = memory.update(
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[0.1, 0.2, 0.3]]]),
            torch.tensor([[[1.0, 2.0]]]),
            torch.tensor([[[False]]]),
        )

        self.assertAlmostEqual(float(memory.usage_count.sum()), 0.0)
        self.assertAlmostEqual(float(stats["write_rate"]), 0.0)

    def test_rule_apply_fuses_prediction_and_memory(self):
        apply = RuleApply(SimpleNamespace(use_memory_fusion=True))
        out = apply(
            rho_t=torch.tensor([[[0.5, 1.0]]]),
            delta_rule_pred=torch.tensor([[[1.0, 1.0]]]),
            memory_delta_rule=torch.tensor([[[0.0, 2.0]]]),
            operator_conf=torch.tensor([[[0.8]]]),
            binding_conf=torch.tensor([[[0.5]]]),
            memory_conf=torch.tensor([[[0.2]]]),
            gate=torch.tensor(1.0),
        )

        expected_alpha = 0.4 / 0.6
        self.assertAlmostEqual(float(out["alpha"].item()), expected_alpha, places=5)
        torch.testing.assert_close(out["delta_rule_fused"], torch.tensor([[[expected_alpha, 2.0 - expected_alpha]]]))
        torch.testing.assert_close(out["rho_next_pred"], torch.tensor([[[0.5 + expected_alpha, 3.0 - expected_alpha]]]))

    def test_rule_apply_gate_zero_blocks_memory_effect(self):
        apply = RuleApply(SimpleNamespace(use_memory_fusion=True))
        out = apply(
            rho_t=torch.tensor([[[0.5, 1.0]]]),
            delta_rule_pred=torch.tensor([[[1.0, 1.0]]]),
            memory_delta_rule=torch.tensor([[[3.0, 4.0]]]),
            operator_conf=torch.tensor([[[0.8]]]),
            binding_conf=torch.tensor([[[0.5]]]),
            memory_conf=torch.tensor([[[0.9]]]),
            gate=torch.tensor([[[0.0]]]),
        )

        torch.testing.assert_close(out["delta_rule_fused"], torch.zeros_like(out["delta_rule_fused"]))
        torch.testing.assert_close(out["rho_next_pred"], torch.tensor([[[0.5, 1.0]]]))

    def test_phase2_memory_write_gate_rejects_low_quality_predictions(self):
        helper = SimpleNamespace(
            phase2_memory_operator_threshold=0.14,
            phase2_memory_binding_threshold=0.30,
            phase2_memory_write_alignment_threshold=0.60,
            phase2_memory_write_apply_error_threshold=0.10,
            phase2_memory_write_delta_threshold=0.001,
        )
        artifact = SimpleNamespace(
            operator_conf=torch.tensor([[[0.9]]]),
            binding_conf=torch.tensor([[[0.8]]]),
            gate=torch.tensor([[[1.0]]]),
            delta_rule_pred=torch.tensor([[[1.0, -1.0]]]),
            rho_next_pred=torch.tensor([[[2.0, 0.0]]]),
        )
        structured = {
            "event_target": torch.tensor([[[1.0]]]),
            "transition_valid_ratio": torch.tensor([[[1.0]]]),
            "target_delta_rho": torch.tensor([[[-1.0, 1.0]]]),
            "nxt": {"rho_t": torch.tensor([[[0.0, 2.0]]])},
        }

        write_info = Dreamer._phase2_memory_write_mask(helper, artifact, structured)

        self.assertFalse(bool(write_info["quality_mask"].item()))
        self.assertFalse(bool(write_info["write_mask"].item()))

    def test_phase2_memory_write_gate_accepts_high_quality_predictions(self):
        helper = SimpleNamespace(
            phase2_memory_operator_threshold=0.14,
            phase2_memory_binding_threshold=0.30,
            phase2_memory_write_alignment_threshold=0.60,
            phase2_memory_write_apply_error_threshold=0.10,
            phase2_memory_write_delta_threshold=0.001,
        )
        artifact = SimpleNamespace(
            operator_conf=torch.tensor([[[0.9]]]),
            binding_conf=torch.tensor([[[0.8]]]),
            gate=torch.tensor([[[1.0]]]),
            delta_rule_pred=torch.tensor([[[1.0, 0.0]]]),
            rho_next_pred=torch.tensor([[[1.0, 0.0]]]),
        )
        structured = {
            "event_target": torch.tensor([[[1.0]]]),
            "transition_valid_ratio": torch.tensor([[[1.0]]]),
            "target_delta_rho": torch.tensor([[[1.0, 0.0]]]),
            "nxt": {"rho_t": torch.tensor([[[1.0, 0.0]]])},
        }

        write_info = Dreamer._phase2_memory_write_mask(helper, artifact, structured)

        self.assertTrue(bool(write_info["quality_mask"].item()))
        self.assertTrue(bool(write_info["write_mask"].item()))


if __name__ == "__main__":
    unittest.main()
