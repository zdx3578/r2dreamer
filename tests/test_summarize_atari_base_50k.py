import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_atari_base_50k import summarize_run


class SummarizeAtariBase50kTest(unittest.TestCase):
    def test_summarize_run_ignores_step_zero_eval_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            rows = [
                {"step": 0, "episode/eval_score": 999.0},
                {"step": 5000, "episode/eval_score": 10.0},
                {"step": 45000, "episode/eval_score": 20.0},
                {"step": 50000, "episode/eval_score": 30.0},
            ]
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            summary = summarize_run(metrics_path)

            self.assertEqual(summary["eval_count"], 3)
            self.assertEqual(summary["last_eval_step"], 50000)
            self.assertEqual(summary["last_eval_score"], 30.0)
            self.assertEqual(summary["best_eval_score"], 30.0)
            self.assertEqual(summary["eval_last3_mean"], 20.0)
            self.assertEqual(summary["eval_40kplus_mean"], 25.0)


if __name__ == "__main__":
    unittest.main()
