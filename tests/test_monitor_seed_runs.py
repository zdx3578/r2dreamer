import json
import tempfile
import unittest
from pathlib import Path

from scripts.monitor_seed_runs import _metric_peaks, _write_final_summary


class MonitorSeedRunsTest(unittest.TestCase):
    def test_metric_peaks_returns_best_value_and_step(self):
        records = [
            {"step": 100, "train/phase1b/slot_match": 0.21, "train/phase1b/slot_match_random": 0.25, "train/ret": 0.1},
            {"step": 200, "train/phase1b/slot_match": 0.34, "train/phase1b/slot_match_random": 0.31, "train/ret": 0.3},
            {"step": 300, "train/phase1b/slot_match": 0.29, "train/phase1b/slot_match_random": 0.36, "train/ret": 0.2},
        ]

        peaks = _metric_peaks(records)

        self.assertEqual(peaks["slot_match"]["step"], 200)
        self.assertAlmostEqual(peaks["slot_match"]["value"], 0.34)
        self.assertEqual(peaks["slot_match_random"]["step"], 300)
        self.assertAlmostEqual(peaks["slot_match_random"]["value"], 0.36)
        self.assertEqual(peaks["ret"]["step"], 200)
        self.assertAlmostEqual(peaks["ret"]["value"], 0.3)

    def test_write_final_summary_persists_seed_and_overall_peaks(self):
        records = [
            {
                "step": 100,
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
                "train/phase1b/m_obj": 0.21,
                "train/phase1b/slot_match": 0.28,
                "train/phase1b/slot_match_random": 0.18,
                "train/phase1b/slot_match_margin": 0.10,
                "train/phase1b/slot_match_margin_score": 0.12,
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
                "train/ret": 0.4,
            },
            {
                "step": 200,
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
                "train/phase1b/m_obj": 0.24,
                "train/phase1b/slot_match": 0.31,
                "train/phase1b/slot_match_random": 0.18,
                "train/phase1b/slot_match_margin": 0.13,
                "train/phase1b/slot_match_margin_score": 0.16,
                "train/phase1b/slot_cycle": 0.85,
                "train/phase1b/slot_identity": 0.37,
                "train/phase1b/slot_concentration": 0.24,
                "train/phase1b/object_interface": 0.52,
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
                "train/phase2/gate_scale": 0.3,
                "train/phase2/match_gate_scale": 0.5,
                "train/ret": 0.5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_dir = root / "seed_0"
            seed_dir.mkdir()
            metrics_path = seed_dir / "metrics.jsonl"
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

            _write_final_summary(root)

            seed_summary = json.loads((seed_dir / "final_summary.json").read_text(encoding="utf-8"))
            overall_summary = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(seed_summary["peaks"]["slot_match"]["step"], 200)
        self.assertAlmostEqual(seed_summary["peaks"]["slot_match"]["value"], 0.31)
        self.assertEqual(seed_summary["peaks"]["slot_match_random"]["step"], 100)
        self.assertAlmostEqual(seed_summary["peaks"]["slot_match_random"]["value"], 0.18)
        self.assertEqual(seed_summary["peaks"]["slot_match_margin_score"]["step"], 200)
        self.assertAlmostEqual(seed_summary["peaks"]["slot_match_margin_score"]["value"], 0.16)
        self.assertEqual(overall_summary["best_peaks"]["slot_match"]["seed"], "seed_0")
        self.assertEqual(overall_summary["best_peaks"]["slot_match"]["step"], 200)
        self.assertAlmostEqual(overall_summary["peak_value_avg"]["slot_match"], 0.31)


if __name__ == "__main__":
    unittest.main()
