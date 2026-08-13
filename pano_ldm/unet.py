import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


def normalize(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class QKVAttentionLegacy(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.n_heads = heads

    def forward(self, qkv):
        batch, width, length = qkv.shape
        channels = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(batch * self.n_heads, 3 * channels, length).split(
            channels, dim=1
        )
        scale = 1 / math.sqrt(math.sqrt(channels))
        weights = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weights = weights.float().softmax(dim=-1).to(weights.dtype)
        attended = torch.einsum("bts,bcs->bct", weights, v)
        return attended.reshape(batch, -1, length)


class QKVAttentionSDPA(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.n_heads = heads

    def forward(self, qkv):
        batch, width, length = qkv.shape
        channels = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(batch, self.n_heads, 3, channels, length).unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            q.transpose(-1, -2),
            k.transpose(-1, -2),
            v.transpose(-1, -2),
        )
        return attended.transpose(-1, -2).reshape(batch, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads, use_checkpoint=True, use_sdpa=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = normalize(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.attention = (
            QKVAttentionSDPA(heads) if use_sdpa else QKVAttentionLegacy(heads)
        )
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        batch, channels, *spatial = x.shape
        flat = x.reshape(batch, channels, -1)
        h = self.attention(self.qkv(self.norm(flat)))
        return (flat + self.proj_out(h)).reshape(batch, channels, *spatial)


class LatentHDRGuidanceLayer(nn.Module):
    def __init__(self, hdr_channels, out_channels):
        super().__init__()
        hidden_channels = min(max(out_channels // 4, hdr_channels), 128)
        self.net = nn.Sequential(
            nn.Conv2d(hdr_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(hidden_channels, out_channels, 3, padding=1)),
        )

    def forward(self, hdr_z, target):
        hdr_z = hdr_z.to(device=target.device, dtype=target.dtype)
        if hdr_z.shape[-2:] != target.shape[-2:]:
            hdr_z = F.interpolate(
                hdr_z,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.net(hdr_z)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.op = (
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            if use_conv
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.op(x)


class SplitInputConv(nn.Module):
    def __init__(self, latent_channels, condition_channels, out_channels):
        super().__init__()
        self.base = nn.Conv2d(latent_channels, out_channels, 3, padding=1)
        self.condition = zero_module(
            nn.Conv2d(condition_channels, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, x):
        split = self.base.in_channels
        return self.base(x[:, :split]) + self.condition(x[:, split:])


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        use_checkpoint=True,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down
        self.use_checkpoint = use_checkpoint
        self.in_layers = nn.Sequential(
            normalize(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip_connection = (
            nn.Identity()
            if self.out_channels == channels
            else nn.Conv2d(channels, self.out_channels, 1)
        )

    def forward(self, x, emb):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            h = self.in_layers[:-1](x)
            h = self.in_layers[-1](self.h_upd(h))
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)
        emb = self.emb_layers(emb).to(h.dtype)
        while emb.ndim < h.ndim:
            emb = emb[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = self.out_layers(h + emb)
        return self.skip_connection(x) + h


class PanoUNet(nn.Module):
    def __init__(
        self,
        config,
        latent_channels,
        hdr_channels,
    ):
        super().__init__()
        in_channels = 3 * latent_channels + 1
        model_channels = config["model_channels"]
        channel_mult = config["channel_mult"]
        num_res_blocks = config["num_res_blocks"]
        attention_resolutions = config["attention_resolutions"]
        dropout = config["dropout"]
        num_heads = config.get("num_heads")
        num_head_channels = config.get("num_head_channels")
        if (num_heads is None) == (num_head_channels is None):
            raise ValueError("configure exactly one of num_heads and num_head_channels")

        def attention_heads(channels):
            if num_heads is not None:
                return num_heads
            if channels % num_head_channels:
                raise ValueError(
                    f"attention channels {channels} are not divisible by "
                    f"num_head_channels {num_head_channels}"
                )
            return channels // num_head_channels

        use_scale_shift_norm = config["use_scale_shift_norm"]
        resblock_updown = config["resblock_updown"]
        use_checkpoint = config.get("use_checkpoint", True)
        use_sdpa = config.get("use_sdpa", False)
        time_embed_dim = 4 * model_channels

        self.model_channels = model_channels
        self.dtype = torch.float32
        self.hdr_guidance_scale = config.get("hdr_guidance_scale", 0.0)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        channels = model_channels * channel_mult[0]
        input_layer = SplitInputConv(
            latent_channels,
            in_channels - latent_channels,
            channels,
        )
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(input_layer)])
        input_block_channels = [channels]
        downsample = 1
        for level, multiplier in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels,
                        time_embed_dim,
                        dropout,
                        multiplier * model_channels,
                        use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                channels = multiplier * model_channels
                if downsample in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels,
                            attention_heads(channels),
                            use_checkpoint=use_checkpoint,
                            use_sdpa=use_sdpa,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            if level != len(channel_mult) - 1:
                down = (
                    ResBlock(
                        channels,
                        time_embed_dim,
                        dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        use_checkpoint=use_checkpoint,
                    )
                    if resblock_updown
                    else Downsample(channels, True)
                )
                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_channels.append(channels)
                downsample *= 2

        hdr_input_block_channels = list(input_block_channels)
        middle_block_channels = channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint=use_checkpoint,
            ),
            AttentionBlock(
                channels,
                attention_heads(channels),
                use_checkpoint=use_checkpoint,
                use_sdpa=use_sdpa,
            ),
            ResBlock(
                channels,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint=use_checkpoint,
            ),
        )
        self.output_blocks = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(channel_mult))):
            for block_index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ResBlock(
                        channels + skip_channels,
                        time_embed_dim,
                        dropout,
                        model_channels * multiplier,
                        use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                channels = model_channels * multiplier
                if downsample in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels,
                            attention_heads(channels),
                            use_checkpoint=use_checkpoint,
                            use_sdpa=use_sdpa,
                        )
                    )
                if level and block_index == num_res_blocks:
                    up = (
                        ResBlock(
                            channels,
                            time_embed_dim,
                            dropout,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_checkpoint=use_checkpoint,
                        )
                        if resblock_updown
                        else Upsample(channels, True)
                    )
                    layers.append(up)
                    downsample //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalize(channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, latent_channels, 3, padding=1)),
        )
        self.hdr_input_blocks = nn.ModuleList(
            [
                LatentHDRGuidanceLayer(hdr_channels, channels)
                for channels in hdr_input_block_channels
            ]
        )
        self.hdr_middle_block = LatentHDRGuidanceLayer(
            hdr_channels,
            middle_block_channels,
        )

    def apply_hdr_guidance(self, h, hdr_z, layer):
        return h + self.hdr_guidance_scale * layer(hdr_z, h)

    def forward(self, x, timesteps, hdr_z):
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        states = []
        h = x.to(self.dtype)
        for index, module in enumerate(self.input_blocks):
            h = module(h, embedding)
            h = self.apply_hdr_guidance(
                h,
                hdr_z,
                self.hdr_input_blocks[index],
            )
            states.append(h)
        h = self.middle_block(h, embedding)
        h = self.apply_hdr_guidance(h, hdr_z, self.hdr_middle_block)
        for module in self.output_blocks:
            h = module(torch.cat((h, states.pop()), dim=1), embedding)
        return self.out(h).to(x.dtype)
