import torch
from torch import nn


def swish(x):
    return x * torch.sigmoid(x)


def normalize(channels):
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class VectorQuantizer2(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1 / n_e, 1 / n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z.reshape(-1, self.e_dim)
        distances = (
            flat_z.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2 * flat_z @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(z.shape)
        loss = (quantized.detach() - z).square().mean()
        loss = loss + self.beta * (quantized - z.detach()).square().mean()
        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, (None, None, indices)


class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = self.logvar.clamp(-30, 20)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        noise = torch.randn(self.mean.shape, device=self.mean.device)
        return self.mean + self.std * noise

    def mode(self):
        return self.mean

    def kl(self):
        return 0.5 * torch.sum(
            self.mean.square() + self.var - 1 - self.logvar,
            dim=(1, 2, 3),
        )


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, temb=None):
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        weights = torch.bmm(q, k) * channels**-0.5
        weights = weights.softmax(dim=2).permute(0, 2, 1)
        v = v.reshape(batch, channels, height * width)
        h = torch.bmm(v, weights).reshape(batch, channels, height, width)
        return x + self.proj_out(h)


def make_attention(config, channels):
    attention_type = config.get("attn_type", "vanilla")
    if attention_type == "vanilla":
        return AttnBlock(channels)
    if attention_type == "none":
        return nn.Identity()
    raise ValueError(f"unknown autoencoder attention type: {attention_type!r}")


class Encoder(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        ch = config["ch"]
        ch_mult = config["ch_mult"]
        num_res_blocks = config["num_res_blocks"]
        dropout = config["dropout"]
        num_resolutions = len(ch_mult)
        self.conv_in = nn.Conv2d(channels, ch, 3, padding=1)
        in_ch_mult = (1, *ch_mult)
        self.down = nn.ModuleList()
        for level in range(num_resolutions):
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(num_res_blocks):
                down.block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
            if level != num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = make_attention(config, block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        self.norm_out = normalize(block_in)
        output_channels = config["z_channels"]
        if config.get("double_z", True):
            output_channels *= 2
        self.conv_out = nn.Conv2d(block_in, output_channels, 3, padding=1)

    def forward(self, x):
        states = [self.conv_in(x)]
        for level, down in enumerate(self.down):
            for block in down.block:
                states.append(block(states[-1]))
            if level != len(self.down) - 1:
                states.append(down.downsample(states[-1]))
        h = self.mid.block_1(states[-1])
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        return self.conv_out(swish(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        ch = config["ch"]
        ch_mult = config["ch_mult"]
        num_res_blocks = config["num_res_blocks"]
        dropout = config["dropout"]
        num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(config["z_channels"], block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = make_attention(config, block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            block_out = ch * ch_mult[level]
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                up.block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
            if level:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, channels, 3, padding=1)

    def forward(self, z):
        h = self.mid.block_1(self.conv_in(z))
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for level in reversed(range(len(self.up))):
            for block in self.up[level].block:
                h = block(h)
            if level:
                h = self.up[level].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))


class AutoencoderKL(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.encoder = Encoder(config, channels)
        self.decoder = Decoder(config, channels)
        embed_dim = config["embed_dim"]
        z_channels = config["z_channels"]
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior


class VQModelInterface(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.encoder = Encoder(config, channels)
        self.decoder = Decoder(config, channels)
        embed_dim = config["embed_dim"]
        z_channels = config["z_channels"]
        self.quantize = VectorQuantizer2(config["n_embed"], embed_dim)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        return self.quant_conv(self.encoder(x))

    def decode(self, z):
        quantized, _, _ = self.quantize(z)
        return self.decoder(self.post_quant_conv(quantized))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def create_autoencoder(config, channels):
    config = dict(config)
    autoencoder_type = config.get("type", "kl")
    if autoencoder_type == "kl":
        if not config.get("double_z", True):
            raise ValueError("KL autoencoder requires double_z=true")
        config["double_z"] = True
        return AutoencoderKL(config, channels)
    if autoencoder_type == "vq":
        if config.get("double_z", False):
            raise ValueError("VQ autoencoder requires double_z=false")
        config["double_z"] = False
        return VQModelInterface(config, channels)
    raise ValueError(f"unknown autoencoder type: {autoencoder_type!r}")
