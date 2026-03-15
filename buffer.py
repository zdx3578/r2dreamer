import torch
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler, SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.max_size = int(config.max_size)
        self.prioritized = bool(getattr(config, "prioritized", False))
        self.priority_eps = float(getattr(config, "priority_eps", 1e-6))
        self.num_eps = 0
        if self.prioritized:
            sampler = PrioritizedSliceSampler(
                max_capacity=self.max_size,
                alpha=float(getattr(config, "priority_alpha", 0.6)),
                beta=float(getattr(config, "priority_beta", 0.4)),
                eps=self.priority_eps,
                reduction=str(getattr(config, "priority_reduction", "max")),
                num_slices=self.batch_size,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
                strict_length=True,
            )
        else:
            sampler = SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            )
        self._sampler = sampler
        self._buffer = self._make_buffer(self.storage_device)

    def _make_buffer(self, storage_device):
        return ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.max_size, device=storage_device, ndim=2),
            sampler=self._sampler,
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        data = data.unsqueeze(1)
        try:
            self._buffer.extend(data)
        except torch.OutOfMemoryError:
            if self.storage_device.type != "cuda":
                raise
            print(
                "Replay buffer GPU allocation failed; recreating storage on CPU. "
                "Training will continue with slower replay sampling."
            )
            self.storage_device = torch.device("cpu")
            self._buffer = self._make_buffer(self.storage_device)
            self._buffer.extend(data.cpu())

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter, priority=None):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        self._buffer[index[1], index[0]].set_("stoch", stoch)
        self._buffer[index[1], index[0]].set_("deter", deter)
        if self.prioritized and priority is not None:
            flat_priority = priority.reshape(-1).detach().to(torch.float32).clamp_min(self.priority_eps)
            priority_index = torch.stack([index[0], index[1]], dim=-1)
            self._buffer.update_priority(priority_index, flat_priority)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()

    def state_dict(self):
        return {
            "device": str(self.device),
            "storage_device": str(self.storage_device),
            "num_eps": int(self.num_eps),
            "buffer": self._buffer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        storage_device = torch.device(state_dict.get("storage_device", str(self.storage_device)))
        if storage_device != self.storage_device:
            self.storage_device = storage_device
            self._buffer = self._make_buffer(self.storage_device)
        self.num_eps = int(state_dict.get("num_eps", 0))
        self._buffer.load_state_dict(state_dict["buffer"])
