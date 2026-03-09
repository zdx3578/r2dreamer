import tempfile
import unittest

from utils.phase_gates import evaluate_phase1a_gate, evaluate_phase1b_gate, evaluate_phase2_gate, load_metrics_records


class PhaseGateTest(unittest.TestCase):
    def test_phase1a_gate_passes_on_structured_metrics(self):
        records = [
            {
                "train/loss/delta_map": 0.4,
                "train/loss/delta_obj": 0.5,
                "train/loss/delta_global": 0.3,
                "train/loss/event": 0.7,
                "train/phase1a/map_std": 0.2,
                "train/phase1a/obj_std": 0.3,
                "train/phase1a/delta_map_abs": 0.1,
                "train/phase1a/delta_obj_abs": 0.1,
                "train/opt/loss": 12.0,
            }
        ]
        result = evaluate_phase1a_gate(records)
        self.assertTrue(result["ready"])

    def test_phase1b_gate_accepts_rising_m_obj(self):
        records = [
            {
                "train/loss/delta_map": 0.4,
                "train/loss/delta_obj": 0.5,
                "train/loss/delta_global": 0.3,
                "train/loss/event": 0.7,
                "train/phase1a/map_std": 0.2,
                "train/phase1a/obj_std": 0.3,
                "train/phase1a/delta_map_abs": 0.1,
                "train/phase1a/delta_obj_abs": 0.1,
                "train/opt/loss": 12.0,
                "train/loss/obj_stable": 1.2,
                "train/loss/obj_local": 1.0,
                "train/loss/obj_rel": 2.2,
                "train/phase1b/m_obj": 0.20 + 0.01 * idx,
                "train/phase1b/slot_match": 0.28,
                "train/phase1b/slot_match_random": 0.18,
                "train/phase1b/slot_match_margin": 0.10,
                "train/phase1b/slot_cycle": 0.82,
                "train/phase1b/slot_identity": 0.34,
                "train/phase1b/slot_concentration": 0.22,
                "train/phase1b/object_interface": 0.48,
                "train/phase1b/motif_entropy": 0.9,
            }
            for idx in range(5)
        ]
        result = evaluate_phase1b_gate(records, slot_count=8)
        self.assertTrue(result["ready"])
        self.assertGreater(result["summary"]["slot_match_mean"], result["summary"]["random_slot_baseline"])
        self.assertGreater(result["summary"]["slot_match_margin_mean"], 0.02)

    def test_phase1b_gate_accepts_stable_plateau(self):
        values = [0.31, 0.312, 0.311, 0.31, 0.309]
        records = [
            {
                "train/loss/delta_map": 0.4,
                "train/loss/delta_obj": 0.5,
                "train/loss/delta_global": 0.3,
                "train/loss/event": 0.7,
                "train/phase1a/map_std": 0.2,
                "train/phase1a/obj_std": 0.3,
                "train/phase1a/delta_map_abs": 0.1,
                "train/phase1a/delta_obj_abs": 0.1,
                "train/opt/loss": 12.0,
                "train/loss/obj_stable": 1.1,
                "train/loss/obj_local": 1.0,
                "train/loss/obj_rel": 2.2,
                "train/phase1b/m_obj": value,
                "train/phase1b/slot_match": 0.27,
                "train/phase1b/slot_match_random": 0.18,
                "train/phase1b/slot_match_margin": 0.09,
                "train/phase1b/slot_cycle": 0.79,
                "train/phase1b/slot_identity": 0.31,
                "train/phase1b/slot_concentration": 0.20,
                "train/phase1b/object_interface": 0.44,
                "train/phase1b/motif_entropy": 0.9,
            }
            for value in values
        ]
        result = evaluate_phase1b_gate(records, slot_count=8)
        self.assertTrue(result["ready"])
        self.assertGreater(result["summary"]["slot_cycle_mean"], 0.5)
        self.assertGreater(result["summary"]["slot_identity_mean"], 0.2)

    def test_phase1b_gate_uses_empirical_random_baseline(self):
        records = [
            {
                "train/loss/delta_map": 0.4,
                "train/loss/delta_obj": 0.5,
                "train/loss/delta_global": 0.3,
                "train/loss/event": 0.7,
                "train/phase1a/map_std": 0.2,
                "train/phase1a/obj_std": 0.3,
                "train/phase1a/delta_map_abs": 0.1,
                "train/phase1a/delta_obj_abs": 0.1,
                "train/opt/loss": 12.0,
                "train/loss/obj_stable": 1.1,
                "train/loss/obj_local": 1.0,
                "train/loss/obj_rel": 2.2,
                "train/phase1b/m_obj": 0.32,
                "train/phase1b/slot_match": 0.19,
                "train/phase1b/slot_match_random": 0.18,
                "train/phase1b/slot_match_margin": 0.01,
                "train/phase1b/slot_cycle": 0.82,
                "train/phase1b/slot_identity": 0.34,
                "train/phase1b/slot_concentration": 0.22,
                "train/phase1b/object_interface": 0.40,
                "train/phase1b/motif_entropy": 0.9,
            }
            for _ in range(5)
        ]
        result = evaluate_phase1b_gate(records, slot_count=8)
        self.assertFalse(result["ready"])
        self.assertAlmostEqual(result["summary"]["random_slot_baseline"], 0.18)

    def test_load_metrics_records(self):
        content = '\n'.join(['{"step": 1, "train/loss/delta_map": 0.1}', '{"step": 2, "train/loss/delta_map": 0.2}'])
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            records = load_metrics_records(handle.name)
        self.assertEqual(len(records), 2)

    def test_phase2_gate_passes_on_noncollapsed_metrics(self):
        records = [
            {
                "train/loss/delta_map": 0.4,
                "train/loss/delta_obj": 0.5,
                "train/loss/delta_global": 0.3,
                "train/loss/event": 0.7,
                "train/phase1a/map_std": 0.2,
                "train/phase1a/obj_std": 0.3,
                "train/phase1a/delta_map_abs": 0.1,
                "train/phase1a/delta_obj_abs": 0.1,
                "train/opt/loss": 12.0,
                "train/loss/obj_stable": 1.2,
                "train/loss/obj_local": 1.0,
                "train/loss/obj_rel": 2.2,
                "train/phase1b/m_obj": 0.20 + 0.01 * idx,
                "train/phase1b/slot_match": 0.28,
                "train/phase1b/slot_match_random": 0.18,
                "train/phase1b/slot_match_margin": 0.10,
                "train/phase1b/slot_cycle": 0.82,
                "train/phase1b/slot_identity": 0.34,
                "train/phase1b/slot_concentration": 0.22,
                "train/phase1b/object_interface": 0.48,
                "train/phase1b/motif_entropy": 0.9,
                "train/loss/op_assign": 0.4,
                "train/loss/op_proto": 0.3,
                "train/loss/op_reuse": 0.2,
                "train/loss/bind_ce": 0.4,
                "train/loss/bind_consistency": 0.2,
                "train/loss/sig_scope": 0.2,
                "train/loss/sig_duration": 0.2,
                "train/loss/sig_impact": 0.2,
                "train/loss/rule_update": 0.1,
                "train/phase2/operator_entropy": 0.6,
                "train/phase2/operator_usage_entropy": 0.7,
                "train/phase2/binding_entropy": 0.5,
                "train/phase2/signature_std": 0.3,
                "train/phase2/rule_delta_abs": 0.1,
                "train/phase2/gate_scale": 0.2,
                "train/phase2/match_gate_scale": 0.4,
            }
            for idx in range(5)
        ]
        result = evaluate_phase2_gate(records, slot_count=8)
        self.assertTrue(result["ready"])


if __name__ == "__main__":
    unittest.main()
