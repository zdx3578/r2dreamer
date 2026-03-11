from . import parallel, wrappers


def make_parallel_envs(config, env_num):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    return parallel.ParallelEnv(env_constructor, env_num, config.device)


def make_envs(config):
    train_envs = make_parallel_envs(config, config.env_num)
    eval_envs = make_parallel_envs(config, config.eval_episode_num)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    elif suite == "arc3":
        import envs.arc3 as arc3

        env = arc3.Arc3Grid(
            task,
            size=tuple(config.size),
            grid_encoding=config.grid_encoding,
            num_colors=config.num_colors,
            num_special_tokens=getattr(config, "num_special_tokens", 1),
            reward_per_level=config.reward_per_level,
            reward_win=config.reward_win,
            reward_loss=config.reward_loss,
            operation_mode=config.operation_mode,
            environments_dir=config.environments_dir,
            recordings_dir=config.recordings_dir,
            arc_api_key=config.arc_api_key,
            arc_base_url=config.arc_base_url,
            seed=config.seed + id,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
