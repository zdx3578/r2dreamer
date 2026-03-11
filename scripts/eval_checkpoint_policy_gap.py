#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import tools
from dreamer import Dreamer
from envs import make_env
from envs import parallel as parallel_envs


def _round(value):
    return None if value is None else round(float(value), 6)


def _make_eval_envs(env_config):
    def env_constructor(idx):
        return lambda: make_env(env_config, idx)

    return parallel_envs.ParallelEnv(env_constructor, int(env_config.eval_episode_num), env_config.device)


def _close_envs(envs):
    for env in getattr(envs, "envs", []):
        env.close()


@torch.no_grad()
def _evaluate(agent, eval_envs, *, sample_actions):
    done = torch.ones(eval_envs.env_num, dtype=torch.bool, device=agent.device)
    once_done = torch.zeros(eval_envs.env_num, dtype=torch.bool, device=agent.device)
    steps = torch.zeros(eval_envs.env_num, dtype=torch.int32, device=agent.device)
    returns = torch.zeros(eval_envs.env_num, dtype=torch.float32, device=agent.device)
    agent_state = agent.get_initial_state(eval_envs.env_num)
    act = agent_state["prev_action"].clone()

    while not once_done.all():
        steps += ~done * ~once_done
        trans_cpu, done_cpu = eval_envs.step(act.detach().to("cpu"), done.detach().to("cpu"))
        trans = trans_cpu.to(agent.device, non_blocking=True)
        done = done_cpu.to(agent.device)
        trans["action"] = act
        act, agent_state = agent.act(trans, agent_state, eval=not sample_actions)
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done

    return {
        "score_mean": float(returns.mean().item()),
        "length_mean": float(steps.to(torch.float32).mean().item()),
        "episode_scores": [float(x) for x in returns.detach().cpu().tolist()],
        "episode_lengths": [int(x) for x in steps.detach().cpu().tolist()],
    }


def _load_config(config_path, device, eval_episodes):
    config = OmegaConf.load(config_path)
    if device is not None:
        config.device = device
        config.env.device = device
        config.model.device = device
    if eval_episodes is not None:
        config.env.eval_episode_num = int(eval_episodes)
        config.trainer.eval_episode_num = int(eval_episodes)
    config.model.compile = False
    return config


def _build_agent(config, checkpoint_path):
    eval_envs = _make_eval_envs(config.env)
    try:
        agent = Dreamer(config.model, eval_envs.observation_space, eval_envs.action_space).to(config.device)
    finally:
        _close_envs(eval_envs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.to(config.device)
    agent.clone_and_freeze()
    agent.eval()
    return agent


def _run_policy(agent, config, *, sample_actions, repeats, base_seed, seed_stride):
    key = "sample" if sample_actions else "mode"
    runs = []
    for repeat_idx in range(repeats):
        seed = int(base_seed + repeat_idx * seed_stride)
        tools.set_seed_everywhere(seed)
        config.env.seed = seed
        eval_envs = _make_eval_envs(config.env)
        try:
            result = _evaluate(agent, eval_envs, sample_actions=sample_actions)
        finally:
            _close_envs(eval_envs)
        result.update({"repeat": repeat_idx, "seed": seed, "policy": key})
        runs.append(result)
    return runs


def _summarize_runs(runs):
    scores = [run["score_mean"] for run in runs]
    lengths = [run["length_mean"] for run in runs]
    return {
        "score_mean": _round(mean(scores)),
        "score_std": _round(pstdev(scores)) if len(scores) > 1 else 0.0,
        "score_min": _round(min(scores)),
        "score_max": _round(max(scores)),
        "length_mean": _round(mean(lengths)),
        "length_std": _round(pstdev(lengths)) if len(lengths) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else checkpoint_path.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, args.device, args.eval_episodes)
    agent = _build_agent(config, checkpoint_path)

    mode_runs = _run_policy(
        agent,
        config,
        sample_actions=False,
        repeats=int(args.repeats),
        base_seed=int(config.seed),
        seed_stride=int(args.seed_stride),
    )
    sample_runs = _run_policy(
        agent,
        config,
        sample_actions=True,
        repeats=int(args.repeats),
        base_seed=int(config.seed),
        seed_stride=int(args.seed_stride),
    )

    mode_summary = _summarize_runs(mode_runs)
    sample_summary = _summarize_runs(sample_runs)
    result = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "device": str(config.device),
        "eval_episodes": int(config.env.eval_episode_num),
        "repeats": int(args.repeats),
        "mode": {
            "summary": mode_summary,
            "runs": mode_runs,
        },
        "sample": {
            "summary": sample_summary,
            "runs": sample_runs,
        },
        "gap": {
            "sample_minus_mode_mean": _round(sample_summary["score_mean"] - mode_summary["score_mean"]),
            "sample_minus_mode_length_mean": _round(sample_summary["length_mean"] - mode_summary["length_mean"]),
        },
    }

    output_path = args.output.expanduser().resolve() if args.output else checkpoint_path.with_name(
        f"{checkpoint_path.stem}_mode_vs_sample_eval.json"
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
