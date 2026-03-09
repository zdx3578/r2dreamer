import torch
from torch import nn

import tools


class Arc3RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Arc3GridEncoder(nn.Module):
    """Token-aware encoder for ARC3 grid observations.

    The ARC3 grid contains discrete color ids. Instead of treating the grid as a
    generic image tensor, this encoder first converts the grid into token ids,
    applies learned token/position embeddings, and only then uses a compact
    convolutional tower to summarize local patterns into a latent vector.
    """

    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, _ = input_shape
        self.num_colors = int(config.num_colors)
        self.num_special_tokens = int(getattr(config, "num_special_tokens", 1))
        self.vocab_size = self.num_colors + self.num_special_tokens
        self.token_dim = int(config.token_dim)
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.kernel_size = int(config.kernel_size)

        self.token_embed = nn.Embedding(self.vocab_size, self.token_dim)
        self.row_embed = nn.Embedding(h, self.token_dim)
        self.col_embed = nn.Embedding(w, self.token_dim)

        blocks = []
        stage_shapes = []
        in_dim = self.token_dim
        padding = self.kernel_size // 2
        for depth in self.depths:
            stage_layers = [nn.Conv2d(in_dim, depth, self.kernel_size, stride=1, padding=padding, bias=True), nn.MaxPool2d(2, 2)]
            if config.norm:
                stage_layers.append(Arc3RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            stage_layers.append(act())
            blocks.append(nn.Sequential(*stage_layers))
            in_dim = depth
            h, w = h // 2, w // 2
            stage_shapes.append((h, w, depth))

        self.out_dim = self.depths[-1] * h * w
        self.readout_stage = max(0, len(stage_shapes) - 2)
        self.spatial_shape = stage_shapes[self.readout_stage]
        self.blocks = nn.ModuleList(blocks)
        self.apply(tools.weight_init_)

    def forward(self, obs, return_spatial=False):
        # (..., H, W, C)
        batch_shape = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:])
        token_ids = self._to_token_ids(x)

        # (B*, H, W, D)
        x = self.token_embed(token_ids)
        row_ids = torch.arange(token_ids.shape[1], device=token_ids.device)
        col_ids = torch.arange(token_ids.shape[2], device=token_ids.device)
        x = x + self.row_embed(row_ids)[None, :, None, :] + self.col_embed(col_ids)[None, None, :, :]

        # (B*, D, H, W)
        x = x.permute(0, 3, 1, 2)
        spatial_readout = None
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == self.readout_stage:
                spatial_readout = x
        spatial = x.permute(0, 2, 3, 1)
        flat = spatial.reshape(spatial.shape[0], -1)
        flat = flat.reshape(*batch_shape, flat.shape[-1])
        if not return_spatial:
            return flat
        if spatial_readout is None:
            spatial_readout = x
        spatial = spatial_readout.permute(0, 2, 3, 1).reshape(*batch_shape, *self.spatial_shape)
        return flat, spatial

    def _to_token_ids(self, grid):
        if grid.shape[-1] > 1:
            token_ids = torch.argmax(grid, dim=-1)
        else:
            token_ids = grid[..., 0]
            if torch.is_floating_point(token_ids):
                token_ids = torch.round(token_ids * float(max(1, self.vocab_size - 1)))
        return token_ids.long().clamp_(0, self.vocab_size - 1)
