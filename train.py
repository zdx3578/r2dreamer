import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs, make_parallel_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.require_python()
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
    probe_eval_envs = None
    sample_eval_envs = None
    if int(getattr(config.trainer, "sample_eval_episode_num", 0)) > 0:
        probe_eval_envs = make_parallel_envs(config.env, int(config.trainer.sample_eval_episode_num))
        sample_eval_envs = make_parallel_envs(config.env, int(config.trainer.sample_eval_episode_num))

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_envs,
        eval_envs,
        probe_eval_envs=probe_eval_envs,
        sample_eval_envs=sample_eval_envs,
    )
    resume_from = getattr(config.trainer, "resume_from", None)
    auto_resume = bool(getattr(config.trainer, "auto_resume", False))
    checkpoint_path = None
    if resume_from:
        checkpoint_path = pathlib.Path(resume_from).expanduser()
    elif auto_resume:
        candidate = logdir / "latest.pt"
        if candidate.exists():
            checkpoint_path = candidate
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        print(f"Resuming from checkpoint {checkpoint_path}")
        policy_trainer.load_checkpoint(agent, checkpoint_path)
    final_step = policy_trainer.begin(agent)
    policy_trainer.save_latest(agent, final_step)


if __name__ == "__main__":
    main()
