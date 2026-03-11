import torch

import tools


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, probe_eval_envs=None, sample_eval_envs=None):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.probe_eval_envs = probe_eval_envs
        self.sample_eval_envs = sample_eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.save_every = int(getattr(config, "save_every", 0))
        self.eval_episode_num = int(config.eval_episode_num)
        self.sample_eval_episode_num = int(getattr(config, "sample_eval_episode_num", 0))
        self.eval_gap_checkpoint_threshold = float(getattr(config, "eval_gap_checkpoint_threshold", 0.0))
        self.eval_drop_checkpoint_ratio = float(getattr(config, "eval_drop_checkpoint_ratio", 0.5))
        self.eval_drop_checkpoint_sample_ratio = float(getattr(config, "eval_drop_checkpoint_sample_ratio", 0.75))
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._should_save = tools.Every(self.save_every)
        self._action_repeat = config.action_repeat
        self._checkpoint_dir = None
        self._prev_eval_score = None
        self._prev_probe_mode_score = None
        if self.logdir is not None:
            self._checkpoint_dir = self.logdir / "checkpoints"
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_items(self, agent, step):
        return {
            "step": int(step),
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }

    def save_latest(self, agent, step):
        if self.logdir is None:
            return
        torch.save(self._checkpoint_items(agent, step), self.logdir / "latest.pt")

    def save_snapshot(self, agent, step):
        if self._checkpoint_dir is None or step <= 0:
            return
        checkpoint_path = self._checkpoint_dir / f"step_{int(step):08d}.pt"
        torch.save(self._checkpoint_items(agent, step), checkpoint_path)

    def save_eval_alert_snapshot(self, agent, step):
        if self._checkpoint_dir is None or step <= 0:
            return
        checkpoint_path = self._checkpoint_dir / f"eval_alert_step_{int(step):08d}.pt"
        torch.save(self._checkpoint_items(agent, step), checkpoint_path)

    def _run_eval_episodes(self, agent, envs, *, deterministic, log_prefix, capture_video=False):
        """Run evaluation episodes for one policy variant.

        Environment stepping is executed on CPU to avoid GPU<->CPU synchronizations
        in the worker processes. Observations are moved back to GPU asynchronously
        (H2D with non_blocking=True) right before policy inference.
        """
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments on CPU.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if capture_video and len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=deterministic)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        return {
            "score": returns.mean(),
            "length": steps.to(torch.float32).mean(),
            "log_metrics": log_metrics,
            "cache": torch.stack(cache, dim=1) if len(cache) else None,
            "log_prefix": log_prefix,
        }

    def _log_eval_result(self, result, *, primary=False):
        score = result["score"]
        length = result["length"]
        log_prefix = result["log_prefix"]
        log_metrics = result["log_metrics"]
        cache = result["cache"]

        if primary:
            self.logger.scalar("episode/eval_score", score)
            self.logger.scalar("episode/eval_length", length)
        self.logger.scalar(f"episode/{log_prefix}_score", score)
        self.logger.scalar(f"episode/{log_prefix}_length", length)
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            if primary:
                self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
            self.logger.scalar(f"episode/{log_prefix}_{key[4:]}", value.mean())
        return cache

    def eval(self, agent, train_step):
        print("Evaluating the policy...")
        agent.eval()

        mode_result = self._run_eval_episodes(
            agent,
            self.eval_envs,
            deterministic=True,
            log_prefix="eval_mode",
            capture_video=True,
        )
        cache = self._log_eval_result(mode_result, primary=True)
        eval_score = float(mode_result["score"].detach().cpu())
        zero_collapse_triggered = 1.0 if self._prev_eval_score is not None and eval_score == 0.0 and self._prev_eval_score > 0 else 0.0
        gap_triggered = 0.0
        split_drop_triggered = 0.0
        checkpoint_saved = 0.0

        if self.sample_eval_episode_num > 0 and self.probe_eval_envs is not None and self.sample_eval_envs is not None:
            probe_mode_result = self._run_eval_episodes(
                agent,
                self.probe_eval_envs,
                deterministic=True,
                log_prefix="eval_probe_mode",
            )
            sample_result = self._run_eval_episodes(
                agent,
                self.sample_eval_envs,
                deterministic=False,
                log_prefix="eval_sample",
            )
            self._log_eval_result(probe_mode_result)
            self._log_eval_result(sample_result)
            gap = sample_result["score"] - probe_mode_result["score"]
            gap_abs = torch.abs(gap)
            self.logger.scalar("episode/eval_gap", gap)
            self.logger.scalar("episode/eval_gap_abs", gap_abs)
            gap_abs_value = float(gap_abs.detach().cpu())
            probe_mode_score = float(probe_mode_result["score"].detach().cpu())
            sample_score = float(sample_result["score"].detach().cpu())
            gap_triggered = 1.0 if self.eval_gap_checkpoint_threshold > 0 and gap_abs_value >= self.eval_gap_checkpoint_threshold else 0.0
            if self._prev_probe_mode_score is not None:
                if (
                    probe_mode_score <= self._prev_probe_mode_score * self.eval_drop_checkpoint_ratio
                    and sample_score >= self._prev_probe_mode_score * self.eval_drop_checkpoint_sample_ratio
                ):
                    split_drop_triggered = 1.0
            self._prev_probe_mode_score = probe_mode_score

        if gap_triggered or zero_collapse_triggered or split_drop_triggered:
            self.save_eval_alert_snapshot(agent, train_step)
            checkpoint_saved = 1.0
        self.logger.scalar("episode/eval_gap_checkpoint_saved", checkpoint_saved)
        self.logger.scalar("episode/eval_zero_collapse_triggered", zero_collapse_triggered)
        if self.sample_eval_episode_num > 0 and self.probe_eval_envs is not None and self.sample_eval_envs is not None:
            self.logger.scalar("episode/eval_gap_triggered", gap_triggered)
            self.logger.scalar("episode/eval_split_drop_triggered", split_drop_triggered)

        self._prev_eval_score = eval_score

        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        The loop is designed to overlap CPU environment stepping and GPU model
        execution. Environments are stepped on CPU, observations are pinned,
        then transferred to GPU with non_blocking=True.
        """
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Increment this to prevent sampling across episode boundaries
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                if step > 0:
                    self.eval(agent, step)
            if self._should_save(step) and step > 0:
                self.save_snapshot(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments on CPU to avoid GPU<->CPU sync in the worker processes.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
        return step
