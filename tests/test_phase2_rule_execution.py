import unittest
from types import SimpleNamespace

import torch

from rule_apply import RuleApply
from rule_memory import RuleMemory


class Phase2RuleExecutionTest(unittest.TestCase):
    def test_rule_memory_update_and_retrieve(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.0, memory_retrieve_temperature=1.0),
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

    def test_rule_memory_keeps_operator_binding_separated(self):
        memory = RuleMemory(
            SimpleNamespace(memory_ema_decay=0.0, memory_retrieve_temperature=1.0),
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
            SimpleNamespace(memory_ema_decay=0.0, memory_retrieve_temperature=1.0),
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


if __name__ == "__main__":
    unittest.main()
