#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


def _safe_mean(values):
    return mean(values) if values else None


def _round(value):
    if value is None:
        return None
    return round(float(value), 6)


def summarize_run(metrics_path: Path):
    train_points = []
    train_episodes = []
    eval_points = []
    with metrics_path.open() as f:
        for line in f:
            row = json.loads(line)
            if "train/ret" in row:
                train_points.append(row)
            if "episode/score" in row:
                train_episodes.append(row)
            if "episode/eval_score" in row and row.get("step", 0) != 0:
                eval_points.append(row)

    last_train = train_points[-1] if train_points else {}
    late_train_episodes = [row["episode/score"] for row in train_episodes if row["step"] >= 40000]
    late_eval_scores = [row["episode/eval_score"] for row in eval_points if row["step"] >= 40000]

    return {
        "metrics_path": str(metrics_path),
        "last_train_step": last_train.get("step"),
        "train_ret_last": _round(last_train.get("train/ret")),
        "train_val_last": _round(last_train.get("train/val")),
        "train_rew_last": _round(last_train.get("train/rew")),
        "train_updates_last": _round(last_train.get("train/opt/updates")),
        "last_train_episode_score": _round(train_episodes[-1]["episode/score"]) if train_episodes else None,
        "last_train_episode_step": train_episodes[-1]["step"] if train_episodes else None,
        "train_episode_max": _round(max((row["episode/score"] for row in train_episodes), default=None)),
        "train_episode_last5_mean": _round(_safe_mean([row["episode/score"] for row in train_episodes[-5:]])),
        "train_episode_40kplus_mean": _round(_safe_mean(late_train_episodes)),
        "train_episode_count": len(train_episodes),
        "last_eval_score": _round(eval_points[-1]["episode/eval_score"]) if eval_points else None,
        "last_eval_step": eval_points[-1]["step"] if eval_points else None,
        "best_eval_score": _round(max((row["episode/eval_score"] for row in eval_points), default=None)),
        "eval_last3_mean": _round(_safe_mean([row["episode/eval_score"] for row in eval_points[-3:]])),
        "eval_40kplus_mean": _round(_safe_mean(late_eval_scores)),
        "eval_count": len(eval_points),
    }

def collect_runs(base_logdir: Path):
    results = {}
    for metrics_path in sorted(base_logdir.glob("*/*/metrics.jsonl")):
        variant = metrics_path.parent.parent.name
        seed = metrics_path.parent.name
        if "_aborted_" in seed:
            continue
        results.setdefault(variant, {})[seed] = summarize_run(metrics_path)
    return results


def render_markdown(results):
    lines = ["# Atari Base 50k Summary", ""]
    for variant in sorted(results):
        lines.append(f"## {variant}")
        lines.append("")
        lines.append("| seed | last_eval | best_eval | eval_last3_mean | eval_40kplus_mean | last_train_episode | last5_train_mean | train_40kplus_mean | train_ret_last | train_val_last |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for seed in sorted(results[variant]):
            row = results[variant][seed]
            lines.append(
                "| {seed} | {last_eval} | {best_eval} | {eval_last3} | {eval_40k} | {last_train_ep} | {last5_train} | {train_40k} | {train_ret} | {train_val} |".format(
                    seed=seed,
                    last_eval=row["last_eval_score"],
                    best_eval=row["best_eval_score"],
                    eval_last3=row["eval_last3_mean"],
                    eval_40k=row["eval_40kplus_mean"],
                    last_train_ep=row["last_train_episode_score"],
                    last5_train=row["train_episode_last5_mean"],
                    train_40k=row["train_episode_40kplus_mean"],
                    train_ret=row["train_ret_last"],
                    train_val=row["train_val_last"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_logdir", type=Path)
    args = parser.parse_args()

    results = collect_runs(args.base_logdir)
    summary_json = args.base_logdir / "summary.json"
    summary_md = args.base_logdir / "summary.md"
    summary_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    summary_md.write_text(render_markdown(results) + "\n")
    print(summary_json)
    print(summary_md)


if __name__ == "__main__":
    main()
