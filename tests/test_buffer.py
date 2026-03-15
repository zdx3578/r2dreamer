import unittest

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from buffer import Buffer


def make_buffer_config(prioritized):
    return OmegaConf.create(
        {
            "device": "cpu",
            "storage_device": "cpu",
            "batch_size": 2,
            "batch_length": 2,
            "max_size": 64,
            "prioritized": prioritized,
            "priority_alpha": 0.6,
            "priority_beta": 0.4,
            "priority_eps": 1e-6,
            "priority_reduction": "max",
        }
    )


def make_transition(step):
    return TensorDict(
        {
            "action": torch.randn(2, 3),
            "stoch": torch.randn(2, 2, 2),
            "deter": torch.randn(2, 4),
            "episode": torch.tensor([0, 1], dtype=torch.int64),
            "reward": torch.tensor([[float(step)], [float(step)]], dtype=torch.float32),
        },
        batch_size=[2],
    )


class BufferTest(unittest.TestCase):
    def test_prioritized_buffer_updates_priority(self):
        replay = Buffer(make_buffer_config(prioritized=True))
        for step in range(6):
            replay.add_transition(make_transition(step))

        data, index, initial = replay.sample()
        self.assertEqual(replay._buffer._sampler.__class__.__name__, "PrioritizedSliceSampler")
        self.assertEqual(initial[0].shape[0], 2)

        priority = torch.linspace(1.0, 2.0, steps=data.shape[0] * data.shape[1], dtype=torch.float32).reshape(
            data.shape[0], data.shape[1]
        )
        replay.update(index, data["stoch"], data["deter"], priority=priority)

        sampled, _, _ = replay.sample()
        self.assertEqual(sampled["stoch"].shape[1], 2)

    def test_uniform_buffer_uses_slice_sampler(self):
        replay = Buffer(make_buffer_config(prioritized=False))
        self.assertEqual(replay._buffer._sampler.__class__.__name__, "SliceSampler")

    def test_buffer_state_dict_roundtrip_restores_count(self):
        replay = Buffer(make_buffer_config(prioritized=False))
        for step in range(4):
            replay.add_transition(make_transition(step))

        restored = Buffer(make_buffer_config(prioritized=False))
        restored.load_state_dict(replay.state_dict())

        self.assertEqual(restored.count(), replay.count())
        self.assertEqual(restored.storage_device.type, replay.storage_device.type)


if __name__ == "__main__":
    unittest.main()
