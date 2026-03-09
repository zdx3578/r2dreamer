import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import binding_head
import objectification
import operator_bank
import phase1a
import rssm
import rule_update
import signature_head
import tools
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.RSSM(
            config.rssm,
            self.embed_size,
            self.act_dim,
        )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)
        self.use_structured_readout = bool(getattr(config, "use_structured_readout", False))
        self.use_effect_model = bool(getattr(config, "use_effect_model", False))
        self.use_reachability_head = bool(getattr(config, "use_reachability_head", False))
        self.use_goal_progress_head = bool(getattr(config, "use_goal_progress_head", False))
        self.use_objectification = bool(getattr(config, "use_objectification", False))
        self.use_operator_bank = bool(getattr(config, "use_operator_bank", False))
        self.use_binding_head = bool(getattr(config, "use_binding_head", False))
        self.use_signature_head = bool(getattr(config, "use_signature_head", False))
        self.use_rule_update = bool(getattr(config, "use_rule_update", False))
        self.goal_horizon = int(getattr(getattr(config, "phase1a", {}), "goal_horizon", 3))
        self.phase2_m_obj_threshold = float(getattr(getattr(config, "phase2", {}), "m_obj_threshold", 0.2))

        if self.use_effect_model and not self.use_structured_readout:
            raise ValueError("Effect model requires use_structured_readout=True.")
        if (self.use_reachability_head or self.use_goal_progress_head) and not self.use_structured_readout:
            raise ValueError("Phase 1A heads require use_structured_readout=True.")
        if self.use_objectification and not (self.use_structured_readout and self.use_effect_model):
            raise ValueError("Objectification requires both structured readout and effect model.")
        if self.use_operator_bank and not (self.use_objectification and self.use_effect_model):
            raise ValueError("Operator bank requires objectification and effect model.")
        if (self.use_binding_head or self.use_signature_head or self.use_rule_update) and not self.use_operator_bank:
            raise ValueError("Binding/signature/rule-update require use_operator_bank=True.")
        if self.use_operator_bank and not (self.use_binding_head and self.use_signature_head and self.use_rule_update):
            raise ValueError("Phase 2 requires operator bank, binding head, signature head, and rule update together.")

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })

        if self.use_structured_readout:
            self.structured_readout = phase1a.StructuredReadout(
                config.structured_readout,
                self.rssm.feat_size,
                config.act,
                bool(config.norm),
            )
            modules.update({"structured_readout": self.structured_readout})
            self._loss_scales.setdefault("struct_map", 1.0)
            self._loss_scales.setdefault("struct_obj", 1.0)
            self._loss_scales.setdefault("struct_global", 1.0)
        if self.use_effect_model:
            self.effect_model = phase1a.EffectModel(
                config.effect_model,
                self.rssm.feat_size,
                self.act_dim,
                self.structured_readout.rule_dim,
                config.act,
                bool(config.norm),
            )
            self.effect_heads = phase1a.EffectHeads(
                config.effect_heads,
                int(config.effect_model.latent_dim),
                self.structured_readout.map_slots,
                self.structured_readout.map_dim,
                self.structured_readout.obj_slots,
                self.structured_readout.obj_dim,
                self.structured_readout.global_dim,
                config.act,
                bool(config.norm),
            )
            modules.update({
                "effect_model": self.effect_model,
                "effect_heads": self.effect_heads,
            })
            self._loss_scales.setdefault("delta_map", 1.0)
            self._loss_scales.setdefault("delta_obj", 1.0)
            self._loss_scales.setdefault("delta_global", 1.0)
            self._loss_scales.setdefault("event", 1.0)
        if self.use_reachability_head:
            self.reachability_head = phase1a.ReachabilityHead(
                config.reachability_head,
                self.rssm.feat_size,
                self.structured_readout.map_slots,
                self.structured_readout.map_dim,
                config.act,
                bool(config.norm),
            )
            modules.update({"reachability_head": self.reachability_head})
            self._loss_scales.setdefault("reach", 1.0)
        if self.use_goal_progress_head:
            self.goal_progress_head = phase1a.GoalProgressHead(
                config.goal_progress_head,
                self.rssm.feat_size,
                self.structured_readout.global_dim,
                config.act,
                bool(config.norm),
            )
            modules.update({"goal_progress_head": self.goal_progress_head})
            self._loss_scales.setdefault("goal", 1.0)
        if self.use_objectification:
            self.objectification = objectification.ObjectificationModule(
                config.objectification,
                self.structured_readout.obj_slots,
                self.structured_readout.obj_dim,
                int(config.effect_model.latent_dim),
            )
            modules.update({"objectification": self.objectification})
            self._loss_scales.setdefault("obj_stable", 1.0)
            self._loss_scales.setdefault("obj_local", 1.0)
            self._loss_scales.setdefault("obj_rel", 1.0)
        if self.use_operator_bank:
            self.operator_bank = operator_bank.OperatorBank(
                config.operator_bank,
                self.rssm.feat_size,
                self.act_dim,
                self.structured_readout.map_slots,
                self.structured_readout.map_dim,
                self.structured_readout.obj_slots,
                self.structured_readout.obj_dim,
                self.structured_readout.global_dim,
                self.structured_readout.rule_dim,
                int(config.effect_model.latent_dim),
                config.act,
                bool(config.norm),
            )
            modules.update({"operator_bank": self.operator_bank})
            self._loss_scales.setdefault("op_assign", 1.0)
            self._loss_scales.setdefault("op_proto", 1.0)
            self._loss_scales.setdefault("op_reuse", 1.0)
        if self.use_binding_head:
            self.binding_head = binding_head.BindingHead(
                config.binding_head,
                int(config.operator_bank.operator_dim),
                int(config.operator_bank.operator_dim),
                config.act,
                bool(config.norm),
            )
            modules.update({"binding_head": self.binding_head})
            self._loss_scales.setdefault("bind_ce", 1.0)
            self._loss_scales.setdefault("bind_consistency", 1.0)
        if self.use_signature_head:
            self.signature_head = signature_head.SignatureHead(
                config.signature_head,
                int(config.operator_bank.operator_dim),
                int(config.operator_bank.operator_dim),
                config.act,
                bool(config.norm),
            )
            modules.update({"signature_head": self.signature_head})
            self._loss_scales.setdefault("sig_scope", 1.0)
            self._loss_scales.setdefault("sig_duration", 1.0)
            self._loss_scales.setdefault("sig_impact", 1.0)
        if self.use_rule_update:
            self.rule_update_head = rule_update.RuleUpdateHead(
                config.rule_update,
                int(config.effect_model.latent_dim),
                int(config.operator_bank.operator_dim),
                int(config.binding_head.num_bindings),
                3,
                self.structured_readout.rule_dim,
                config.act,
                bool(config.norm),
            )
            modules.update({"rule_update_head": self.rule_update_head})
            self._loss_scales.setdefault("rule_update", 1.0)
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler(device=self.device.type, enabled=self.device.type == "cuda")

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            if self.device.type == "cuda":
                print("Compiling update function with torch.compile...")
                self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")
            else:
                print("Skipping torch.compile because current device is not CUDA.")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        # (B, S, K), (B, D)
        stoch, deter, _ = self._frozen_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
        # (B, F)
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(feat)
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_last_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _build_structured_context(self, feat, data, initial):
        initial_feat = self.rssm.get_feat(initial[0], initial[1]).unsqueeze(1)
        feat_seq = torch.cat([initial_feat, feat], dim=1)
        valid_mask = data.get("valid_mask")
        if valid_mask is not None:
            valid_mask_seq = torch.cat([valid_mask[:, :1], valid_mask], dim=1)
        else:
            valid_mask_seq = None
        readouts = self.structured_readout(feat_seq, valid_mask=valid_mask_seq)
        current = {
            "M_t": readouts["M_t"][:, :-1],
            "O_t": readouts["O_t"][:, :-1],
            "g_t": readouts["g_t"][:, :-1],
            "rho_t": readouts["rho_t"][:, :-1],
            "map_mask": readouts["map_mask"][:, :-1],
            "obj_mask": readouts["obj_mask"][:, :-1],
            "valid_ratio": readouts["valid_ratio"][:, :-1],
        }
        nxt = {
            "M_t": readouts["M_t"][:, 1:],
            "O_t": readouts["O_t"][:, 1:],
            "g_t": readouts["g_t"][:, 1:],
            "rho_t": readouts["rho_t"][:, 1:],
            "map_mask": readouts["map_mask"][:, 1:],
            "obj_mask": readouts["obj_mask"][:, 1:],
            "valid_ratio": readouts["valid_ratio"][:, 1:],
        }
        structured = {
            "feat_seq": feat_seq,
            "readouts": readouts,
            "current": current,
            "nxt": nxt,
            "transition_map_mask": (current["map_mask"] * nxt["map_mask"]).detach(),
            "transition_obj_mask": (current["obj_mask"] * nxt["obj_mask"]).detach(),
            "transition_valid_ratio": (current["valid_ratio"] * nxt["valid_ratio"]).detach(),
            "target_delta_g": (nxt["g_t"] - current["g_t"]).detach(),
            "target_delta_rho": (nxt["rho_t"] - current["rho_t"]).detach(),
        }
        if self.use_effect_model:
            target_delta_M = (nxt["M_t"] - current["M_t"]).detach()
            target_delta_O = (nxt["O_t"] - current["O_t"]).detach()
            z_eff = self.effect_model(feat_seq[:, :-1], data["action"], current["rho_t"])
            effect_out = self.effect_heads(z_eff)
            reward_target = self._seq_scalar(data["reward"])
            terminal_target = self._seq_scalar(data["is_terminal"]).bool()
            delta_struct = (
                (target_delta_M.abs() * structured["transition_map_mask"]).mean(dim=(-2, -1))
                + (target_delta_O.abs() * structured["transition_obj_mask"]).mean(dim=(-2, -1))
                + (structured["target_delta_g"].abs() * structured["transition_valid_ratio"]).mean(dim=-1)
            ).unsqueeze(-1)
            threshold = delta_struct.mean().detach() + 0.5 * delta_struct.std(unbiased=False).detach()
            explicit_event = reward_target.abs() > 1e-6
            structural_event = delta_struct > threshold
            event_target = (explicit_event | terminal_target | structural_event).to(effect_out["event_logits"].dtype)
            structured.update(
                {
                    "z_eff": z_eff,
                    "effect_out": effect_out,
                    "target_delta_M": target_delta_M,
                    "target_delta_O": target_delta_O,
                    "event_target": event_target,
                    "reward_target": reward_target,
                    "terminal_target": terminal_target,
                }
            )
        return structured

    def _weighted_loss(self, pred, target, weight=None, kind="mse"):
        if kind == "mse":
            value = F.mse_loss(pred, target, reduction="none")
        elif kind == "smooth_l1":
            value = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(kind)
        if weight is None:
            return value.mean()
        weight = weight.to(value.dtype)
        while weight.dim() < value.dim():
            weight = weight.unsqueeze(-1)
        weight = torch.broadcast_to(weight, value.shape)
        numer = (value * weight).sum()
        denom = weight.sum().clamp_min(1e-6)
        return numer / denom

    def _phase1a_losses(self, structured, data):
        losses = {}
        metrics = {}
        feat_seq = structured["feat_seq"]
        readouts = structured["readouts"]
        current = structured["current"]
        feat_target = feat_seq.detach()
        losses["struct_map"] = self._weighted_loss(readouts["M_recon"], feat_target, readouts["valid_ratio"], kind="mse")
        losses["struct_obj"] = self._weighted_loss(readouts["O_recon"], feat_target, readouts["valid_ratio"], kind="mse")
        losses["struct_global"] = self._weighted_loss(
            readouts["g_rho_recon"], feat_target, readouts["valid_ratio"], kind="mse"
        )

        metrics["phase1a/map_std"] = readouts["M_t"].std()
        metrics["phase1a/obj_std"] = readouts["O_t"].std()
        metrics["phase1a/global_std"] = readouts["g_t"].std()
        metrics["phase1a/rule_std"] = readouts["rho_t"].std()

        target_delta_g = structured["target_delta_g"]
        if self.use_effect_model:
            target_delta_M = structured["target_delta_M"]
            target_delta_O = structured["target_delta_O"]
            effect_out = structured["effect_out"]

            losses["delta_map"] = self._weighted_loss(
                effect_out["delta_map"], target_delta_M, structured["transition_map_mask"], kind="smooth_l1"
            )
            losses["delta_obj"] = self._weighted_loss(
                effect_out["delta_obj"], target_delta_O, structured["transition_obj_mask"], kind="smooth_l1"
            )
            losses["delta_global"] = self._weighted_loss(
                effect_out["delta_global"], target_delta_g, structured["transition_valid_ratio"], kind="smooth_l1"
            )
            event_target = structured["event_target"]
            losses["event"] = F.binary_cross_entropy_with_logits(effect_out["event_logits"], event_target)

            metrics["phase1a/delta_map_abs"] = (target_delta_M.abs() * structured["transition_map_mask"]).mean()
            metrics["phase1a/delta_obj_abs"] = (target_delta_O.abs() * structured["transition_obj_mask"]).mean()
            metrics["phase1a/delta_global_abs"] = (target_delta_g.abs() * structured["transition_valid_ratio"]).mean()
            metrics["phase1a/event_rate"] = event_target.mean()

        feat_delta = (feat_seq[:, 1:] - feat_seq[:, :-1]).detach()
        if self.use_reachability_head:
            reach_pred = self.reachability_head(feat_seq[:, :-1], current["M_t"])
            reach_target = torch.tanh(feat_delta.abs().mean(dim=-1, keepdim=True))
            losses["reach"] = F.smooth_l1_loss(reach_pred, reach_target)
            metrics["phase1a/reach_target"] = reach_target.mean()

        if self.use_goal_progress_head:
            goal_pred = self.goal_progress_head(feat_seq[:, :-1], current["g_t"])
            goal_target = self._short_horizon_return(
                self._seq_scalar(data["reward"]),
                self._seq_scalar(data["is_terminal"]),
                1 - 1 / self.horizon,
                self.goal_horizon,
            )
            goal_target = torch.clamp(
                torch.tanh(goal_target) + 0.25 * torch.tanh(target_delta_g.abs().mean(dim=-1, keepdim=True)),
                -1.0,
                1.0,
            ).detach()
            losses["goal"] = F.smooth_l1_loss(goal_pred, goal_target)
            metrics["phase1a/goal_target"] = goal_target.mean()

        return losses, metrics

    def _objectification_losses(self, structured):
        out = self.objectification(
            structured["current"]["O_t"],
            structured["nxt"]["O_t"],
            structured["z_eff"],
            structured["effect_out"]["delta_obj"],
            structured["event_target"],
            structured["transition_obj_mask"],
        )
        losses = {
            "obj_stable": out["loss_obj_stable"],
            "obj_local": out["loss_obj_local"],
            "obj_rel": out["loss_obj_rel"],
        }
        metrics = {
            "phase1b/m_obj": out["objectness_score"],
            "phase1b/slot_match": out["slot_match_score"],
            "phase1b/slot_concentration": out["slot_concentration"],
            "phase1b/motif_entropy": out["motif_usage_entropy"],
        }
        return losses, metrics, out

    def _phase2_gate(self, objectness_score):
        return torch.clamp(
            (objectness_score.detach() - self.phase2_m_obj_threshold) / max(1e-6, 1.0 - self.phase2_m_obj_threshold),
            0.0,
            1.0,
        )

    def _binding_proxy(self, structured):
        target_delta_M = (structured["target_delta_M"].abs() * structured["transition_map_mask"]).mean(dim=(-2, -1))
        target_delta_O = (structured["target_delta_O"].abs() * structured["transition_obj_mask"]).mean(dim=(-2, -1))
        target_delta_g = (structured["target_delta_g"].abs() * structured["transition_valid_ratio"]).mean(dim=-1)
        target_delta_rho = (structured["target_delta_rho"].abs() * structured["transition_valid_ratio"]).mean(dim=-1)
        rel_change = (
            torch.einsum(
                "...id,...jd->...ij",
                structured["current"]["O_t"],
                structured["current"]["O_t"],
            )
            - torch.einsum(
                "...id,...jd->...ij",
                structured["nxt"]["O_t"],
                structured["nxt"]["O_t"],
            )
        ).abs().mean(dim=(-2, -1))
        slot_mass = (structured["target_delta_O"].abs().mean(dim=-1) * structured["transition_obj_mask"].squeeze(-1)) + 1e-6
        slot_prob = slot_mass / slot_mass.sum(dim=-1, keepdim=True)
        slot_concentration = slot_prob.max(dim=-1).values
        spread = 1.0 - slot_concentration

        scores = torch.stack(
            [
                slot_concentration,
                spread * torch.tanh(target_delta_O),
                torch.tanh(rel_change),
                torch.tanh(target_delta_M),
                torch.tanh(target_delta_g + target_delta_rho + structured["event_target"].squeeze(-1)),
            ],
            dim=-1,
        )
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _signature_proxy(self, structured):
        slot_mass = (structured["target_delta_O"].abs().mean(dim=-1) * structured["transition_obj_mask"].squeeze(-1)) + 1e-6
        slot_prob = slot_mass / slot_mass.sum(dim=-1, keepdim=True)
        slot_concentration = slot_prob.max(dim=-1, keepdim=True).values
        map_mass = (structured["target_delta_M"].abs() * structured["transition_map_mask"]).mean(dim=(-2, -1)).unsqueeze(-1)
        global_mass = (structured["target_delta_g"].abs() * structured["transition_valid_ratio"]).mean(dim=-1, keepdim=True)
        rule_mass = (structured["target_delta_rho"].abs() * structured["transition_valid_ratio"]).mean(dim=-1, keepdim=True)
        reward_mass = structured["reward_target"].abs()
        event = structured["event_target"]
        scope_target = torch.clamp(0.5 * (1.0 - slot_concentration) + 0.5 * torch.tanh(map_mass + global_mass), 0.0, 1.0)
        duration_target = torch.clamp(torch.sigmoid(2.0 * (rule_mass + global_mass + event)), 0.0, 1.0)
        impact_target = torch.tanh(reward_mass + global_mass + rule_mass)
        return scope_target, duration_target, impact_target

    def _phase2_losses(self, structured, data):
        op_out = self.operator_bank(
            structured["feat_seq"][:, :-1],
            data["action"],
            structured["current"]["M_t"],
            structured["current"]["O_t"],
            structured["current"]["g_t"],
            structured["current"]["rho_t"],
            structured["z_eff"],
        )
        bind_out = self.binding_head(op_out["operator_embed"], op_out["context_embed"])
        sig_out = self.signature_head(op_out["operator_embed"], op_out["context_embed"])
        rule_out = self.rule_update_head(structured["z_eff"], op_out["operator_embed"], bind_out["q_b"], sig_out["q_sigma"])

        gate = self._phase2_gate(structured["objectification_out"]["objectness_score"])
        flat_q = op_out["q_u"].reshape(-1, op_out["q_u"].shape[-1])
        flat_eff = op_out["effect_embed"].reshape(-1, op_out["effect_embed"].shape[-1])
        flat_binding = bind_out["q_b"].reshape(-1, bind_out["q_b"].shape[-1])

        op_assign = -(op_out["target_q"].detach() * torch.log(op_out["q_u"] + 1e-6)).sum(dim=-1).mean()
        if flat_q.shape[0] > 1:
            eff_sim = torch.matmul(flat_eff, flat_eff.transpose(0, 1))
            op_sim = torch.matmul(flat_q, flat_q.transpose(0, 1))
            op_proto = F.smooth_l1_loss(op_sim, eff_sim.detach())
        else:
            op_proto = torch.zeros_like(op_assign)
        uniform = torch.full_like(op_out["avg_usage"], 1.0 / op_out["avg_usage"].numel())
        op_reuse = (op_out["avg_usage"] * (torch.log(op_out["avg_usage"] + 1e-6) - torch.log(uniform))).sum()

        binding_target = self._binding_proxy(structured)
        bind_ce = -(binding_target.detach() * torch.log(bind_out["q_b"] + 1e-6)).sum(dim=-1).mean()
        operator_binding = torch.matmul(flat_q.transpose(0, 1), flat_binding) / flat_q.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1e-6)
        bind_expected = torch.matmul(flat_q, operator_binding.detach()).reshape_as(bind_out["q_b"])
        bind_consistency = F.smooth_l1_loss(bind_out["q_b"], bind_expected)

        scope_target, duration_target, impact_target = self._signature_proxy(structured)
        sig_scope = F.smooth_l1_loss(sig_out["scope"], scope_target)
        sig_duration = F.smooth_l1_loss(sig_out["duration"], duration_target)
        sig_impact = F.smooth_l1_loss(sig_out["impact"], impact_target)
        rule_update_loss = self._weighted_loss(
            rule_out["delta_rule"], structured["target_delta_rho"], structured["transition_valid_ratio"], kind="smooth_l1"
        )

        losses = {
            "op_assign": gate * op_assign,
            "op_proto": gate * op_proto,
            "op_reuse": gate * op_reuse,
            "bind_ce": gate * bind_ce,
            "bind_consistency": gate * bind_consistency,
            "sig_scope": gate * sig_scope,
            "sig_duration": gate * sig_duration,
            "sig_impact": gate * sig_impact,
            "rule_update": gate * rule_update_loss,
        }
        metrics = {
            "phase2/gate_scale": gate,
            "phase2/operator_entropy": op_out["sample_entropy"],
            "phase2/operator_usage_entropy": op_out["usage_entropy"],
            "phase2/operator_usage_max": op_out["avg_usage"].max(),
            "phase2/binding_entropy": bind_out["entropy"],
            "phase2/binding_max": bind_out["q_b"].max(dim=-1).values.mean(),
            "phase2/signature_scope": sig_out["scope"].mean(),
            "phase2/signature_duration": sig_out["duration"].mean(),
            "phase2/signature_impact": sig_out["impact"].mean(),
            "phase2/signature_std": sig_out["q_sigma"].std(),
            "phase2/rule_delta_abs": (structured["target_delta_rho"].abs() * structured["transition_valid_ratio"]).mean(),
        }
        return losses, metrics

    def _short_horizon_return(self, reward, terminal, disc, horizon):
        B, T, _ = reward.shape
        target = torch.zeros_like(reward)
        for t in range(T):
            running = torch.zeros(B, 1, device=reward.device, dtype=reward.dtype)
            live = torch.ones_like(running)
            discount = 1.0
            for k in range(int(horizon)):
                idx = t + k
                if idx >= T:
                    break
                running = running + discount * live * reward[:, idx]
                live = live * (1.0 - terminal[:, idx])
                discount *= disc
            target[:, t] = running
        return target

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        reward_target = self._seq_scalar(data["reward"])
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(reward_target))
        cont = 1.0 - self._seq_scalar(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())
        if self.use_structured_readout:
            structured = self._build_structured_context(feat, data, initial)
            phase1a_losses, phase1a_metrics = self._phase1a_losses(structured, data)
            losses.update(phase1a_losses)
            metrics.update(phase1a_metrics)
            if self.use_objectification:
                object_losses, object_metrics, object_out = self._objectification_losses(structured)
                losses.update(object_losses)
                metrics.update(object_metrics)
                structured["objectification_out"] = object_out
            if self.use_operator_bank:
                phase2_losses, phase2_metrics = self._phase2_losses(structured, data)
                losses.update(phase2_losses)
                metrics.update(phase2_metrics)

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B, T, ...) -> (B*T, ...)
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            self._seq_scalar(data["is_last"]),
            self._seq_scalar(data["is_terminal"]),
            self._seq_scalar(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    def _seq_scalar(self, value):
        value = to_f32(value)
        if value.dim() == 2:
            return value.unsqueeze(-1)
        return value

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            action = self._frozen_actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
