from functools import partial
from math import cos, isfinite, pi

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.func import functional_call
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from utils import load_config, stage2_sg_from_params
from pano_ldm.autoencoder import create_autoencoder
from pano_ldm.unet import PanoUNet

EMA_DECAY = 0.999


class DDIMSampler:
    def __init__(self, model):
        self.model = model
        self.schedule_key = None

    def make_schedule(self, steps, eta):
        if not 0 < steps < self.model.num_timesteps:
            raise ValueError(
                f"DDIM steps must be between 1 and {self.model.num_timesteps - 1}"
            )
        timesteps = (
            torch.arange(steps, device=self.model.device, dtype=torch.long)
            * self.model.num_timesteps
            // steps
            + 1
        )
        alphas_cumprod = self.model.alphas_cumprod
        alphas = alphas_cumprod[timesteps]
        alphas_prev = torch.cat((alphas_cumprod[:1], alphas_cumprod[timesteps[:-1]]))
        sigmas = eta * torch.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )
        self.ddim_timesteps = timesteps
        self.ddim_alphas = alphas
        self.ddim_alphas_prev = alphas_prev
        self.ddim_sigmas = sigmas
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1 - alphas)

    @torch.no_grad()
    def sample(
        self,
        steps,
        batch_size,
        shape,
        conditioning,
        eta=0,
        x_T=None,
        verbose=True,
    ):
        schedule_key = (steps, float(eta))
        if schedule_key != self.schedule_key:
            self.make_schedule(steps, eta)
            self.schedule_key = schedule_key
        image = x_T
        if image is None:
            image = torch.randn((batch_size, *shape), device=self.model.device)
        time_range = self.ddim_timesteps.flip(0)
        iterator = tqdm(
            time_range,
            desc="DDIM Sampler",
            total=len(time_range),
            disable=not verbose,
        )
        for reverse_index, step in enumerate(iterator):
            index = len(time_range) - reverse_index - 1
            timestep = torch.full(
                (batch_size,),
                step.item(),
                device=self.model.device,
                dtype=torch.long,
            )
            image = self.step(image, conditioning, timestep, index)
        return image

    def step(self, x, condition, timestep, index):
        predicted_noise = self.model.apply_model(x, timestep, condition)
        batch = x.shape[0]
        values_shape = (batch, 1, 1, 1)
        alpha = self.ddim_alphas[index].expand(values_shape)
        alpha_prev = self.ddim_alphas_prev[index].expand(values_shape)
        sigma = self.ddim_sigmas[index].expand(values_shape)
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alphas[index].expand(
            values_shape
        )
        predicted_start = (x - sqrt_one_minus_alpha * predicted_noise) / alpha.sqrt()
        direction = (1 - alpha_prev - sigma.square()).sqrt() * predicted_noise
        noise = sigma * torch.randn_like(x)
        return alpha_prev.sqrt() * predicted_start + direction + noise


def extract(values, timesteps, shape):
    batch = timesteps.shape[0]
    return values.gather(-1, timesteps).reshape(batch, *((1,) * (len(shape) - 1)))


def disable_train(module, mode=True):
    return module


def freeze(module):
    module.eval()
    module.train = partial(disable_train, module)
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


class DiffusionWrapper(nn.Module):
    def __init__(self, config, latent_channels, hdr_channels):
        super().__init__()
        self.diffusion_model = PanoUNet(
            config,
            latent_channels,
            hdr_channels,
        )

    def forward(self, x, timesteps, condition):
        x = torch.cat((x, *condition["c_concat"]), dim=1)
        hdr_z = torch.cat(condition["c_hdr"], dim=1)
        return self.diffusion_model(x, timesteps, hdr_z)


class PanoVAEDiffusion(L.LightningModule):
    def __init__(
        self,
        diffusion_config,
        unet_config,
        autoencoder_config,
        hdr_autoencoder_config,
    ):
        super().__init__()
        if autoencoder_config.get("type") != "vq":
            raise ValueError("Stage 2 first stage must use a VQ autoencoder")
        if hdr_autoencoder_config.get("type") != "kl":
            raise ValueError("Stage 2 HDR stage must use a KL autoencoder")

        latent_channels = autoencoder_config["embed_dim"]
        hdr_channels = hdr_autoencoder_config["embed_dim"]
        self.first_stage_key = "pano"
        self.perspective_key = "perspective"
        self.mask_key = "known_mask"
        self.asg_key = "asg"
        self.sg_key = "sg"
        self.channels = latent_channels
        self.loss_type = diffusion_config.get("loss_type", "l2")
        if self.loss_type != "l2":
            raise ValueError(
                f'diffusion loss_type must be "l2" (MSE), got {self.loss_type!r}'
            )
        self.fused_optimizer = diffusion_config.get("fused_optimizer", False)
        self.non_blocking_transfer = diffusion_config.get(
            "non_blocking_transfer",
            False,
        )
        self.ema_enabled = False
        self.ema_weights = {}
        self.loaded_ema_weights = None
        self.loaded_checkpoint = False
        self.learning_rate = 0.0
        self.warmup_start = 0.0
        self.warmup_steps = 0
        self.constant_steps = 0
        self.decay_target = 0.0
        self.decay_steps = 0
        self.cond_lr_scale = 2.0
        self.hdr_lr_scale = 2.0
        self.freeze_base = False

        if diffusion_config.get("scale_by_std", False):
            self.register_buffer(
                "scale_factor",
                torch.tensor(diffusion_config.get("scale_factor", 1.0)),
            )
        else:
            self.scale_factor = diffusion_config["scale_factor"]

        self.model = DiffusionWrapper(
            unet_config,
            latent_channels,
            hdr_channels,
        )
        self.register_schedule(diffusion_config)
        self.first_stage_model = freeze(
            create_autoencoder(autoencoder_config, channels=3)
        )
        self.hdr_scale_factor = diffusion_config["hdr_scale_factor"]
        self.hdr_stage_model = freeze(
            create_autoencoder(hdr_autoencoder_config, channels=1)
        )

    def configure_training(self, config):
        learning_rate = config["learning_rate"]
        ema = config.get("ema", True)
        warmup_start = config.get("warmup_start", learning_rate)
        warmup_steps = config.get("warmup_steps", 0)
        constant_steps = config.get("constant_steps", 0)
        decay_target = config.get("decay_target", learning_rate)
        decay_steps = config.get("decay_steps", 0)
        cond_lr_scale = config.get("cond_lr_scale", 2)
        hdr_lr_scale = config.get("hdr_lr_scale", 2)
        freeze_base = config.get("freeze_base", False)

        if not isinstance(ema, bool):
            raise ValueError("ema must be true or false")
        if not isinstance(freeze_base, bool):
            raise ValueError("freeze_base must be true or false")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isfinite(warmup_start) or not 0 <= warmup_start <= learning_rate:
            raise ValueError("warmup_start must be in [0, learning_rate]")
        if not isfinite(decay_target) or not 0 <= decay_target <= learning_rate:
            raise ValueError("decay_target must be in [0, learning_rate]")
        schedule_steps = (warmup_steps, constant_steps, decay_steps)
        if any(
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
            for steps in schedule_steps
        ):
            raise ValueError(
                "warmup_steps, constant_steps, and decay_steps "
                "must be non-negative integers"
            )
        if (
            not isfinite(cond_lr_scale)
            or not isfinite(hdr_lr_scale)
            or cond_lr_scale < 0
            or hdr_lr_scale < 0
        ):
            raise ValueError("cond_lr_scale and hdr_lr_scale must be non-negative")

        self.ema_enabled = ema
        self.learning_rate = learning_rate
        self.warmup_start = warmup_start
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_target = decay_target
        self.decay_steps = decay_steps
        self.cond_lr_scale = cond_lr_scale
        self.hdr_lr_scale = hdr_lr_scale
        self.freeze_base = freeze_base
        for name, parameters in self.diffusion_parameter_groups().items():
            trainable = not freeze_base or name != "base"
            for parameter in parameters:
                parameter.requires_grad = trainable
        print(
            "Training settings: "
            f"ema={self.ema_enabled}, lr={self.learning_rate:g}, "
            f"warmup={self.warmup_start:g}/{self.warmup_steps:d}, "
            f"constant={self.constant_steps:d}, "
            f"decay={self.decay_target:g}/{self.decay_steps:d}, "
            f"condition_lr_scale={self.cond_lr_scale:g}, "
            f"hdr_lr_scale={self.hdr_lr_scale:g}, "
            f"freeze_base={self.freeze_base}"
        )

    def diffusion_parameter_groups(self):
        unet = self.model.diffusion_model
        condition = list(unet.input_blocks[0][0].condition.parameters())
        hdr = list(unet.hdr_input_blocks.parameters()) + list(
            unet.hdr_middle_block.parameters()
        )
        separate = {id(parameter) for parameter in condition}
        separate.update(id(parameter) for parameter in hdr)
        base = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in separate
        ]
        return {"base": base, "condition": condition, "hdr": hdr}

    def model_parameters(self):
        return dict(self.model.named_parameters())

    def trainable_model_parameters(self):
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def initialize_ema(self):
        parameters = self.model_parameters()
        loaded = self.loaded_ema_weights
        if loaded is not None:
            missing = sorted(set(parameters) - set(loaded))
            unexpected = sorted(set(loaded) - set(parameters))
            mismatched = [
                name
                for name in parameters.keys() & loaded.keys()
                if parameters[name].shape != loaded[name].shape
            ]
            if missing or unexpected or mismatched:
                raise RuntimeError(
                    "EMA checkpoint does not match the diffusion model: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}, "
                    f"mismatched={len(mismatched)}"
                )
            self.ema_weights = {
                name: loaded[name]
                .detach()
                .to(device=parameter.device, dtype=parameter.dtype)
                .clone()
                for name, parameter in parameters.items()
            }
            print(f"Restored EMA parameters with decay {EMA_DECAY:g}")
        else:
            self.ema_weights = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            if self.loaded_checkpoint:
                print(
                    "Checkpoint has no EMA state; initialized EMA from online weights"
                )
            else:
                print(f"Initialized EMA parameters with decay {EMA_DECAY:g}")
        self.loaded_ema_weights = None

    @torch.no_grad()
    def update_ema(self):
        if not self.ema_weights:
            self.initialize_ema()
        for name, parameter in self.trainable_model_parameters().items():
            self.ema_weights[name].lerp_(parameter.detach(), 1 - EMA_DECAY)

    def on_fit_start(self):
        if self.ema_enabled:
            self.initialize_ema()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema_enabled:
            self.update_ema()

    def on_save_checkpoint(self, checkpoint):
        if not self.ema_enabled:
            return
        if not self.ema_weights:
            self.initialize_ema()
        checkpoint["ema_decay"] = EMA_DECAY
        checkpoint["ema_state_dict"] = {
            f"model.{name}": value.detach().cpu()
            for name, value in self.ema_weights.items()
        }

    def on_load_checkpoint(self, checkpoint):
        self.loaded_checkpoint = True
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is None or not self.ema_enabled:
            self.loaded_ema_weights = None
            return
        if not isinstance(ema_state, dict):
            raise RuntimeError("EMA checkpoint state must be a dictionary")
        decay = checkpoint.get("ema_decay", EMA_DECAY)
        if decay != EMA_DECAY:
            raise RuntimeError(
                f"EMA decay mismatch: checkpoint={decay:g}, expected={EMA_DECAY:g}"
            )
        prefix = "model."
        invalid = [name for name in ema_state if not name.startswith(prefix)]
        if invalid:
            raise RuntimeError(f"EMA checkpoint contains invalid key: {invalid[0]}")
        self.loaded_ema_weights = {
            name[len(prefix) :]: value for name, value in ema_state.items()
        }

    def register_schedule(self, config):
        timesteps = config["timesteps"]
        start = config["linear_start"] ** 0.5
        end = config["linear_end"] ** 0.5
        betas = (
            torch.linspace(start, end, timesteps, dtype=torch.float64).square().numpy()
        )
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1, alphas_cumprod[:-1])
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        to_tensor = partial(torch.tensor, dtype=torch.float32)

        self.num_timesteps = timesteps
        self.register_buffer("betas", to_tensor(betas))
        self.register_buffer("alphas_cumprod", to_tensor(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_tensor(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_tensor(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_tensor(np.sqrt(1 - alphas_cumprod)),
        )
        self.register_buffer("posterior_variance", to_tensor(posterior_variance))
        level_weights = self.betas.square() / (
            2 * self.posterior_variance * to_tensor(alphas) * (1 - self.alphas_cumprod)
        )
        level_weights[0] = level_weights[1]
        self.register_buffer("lvlb_weights", level_weights, persistent=False)

    def get_input_tensor(self, batch, key=None):
        key = key or self.first_stage_key
        value = batch[key]
        if value.ndim == 3:
            value = value[..., None]
        return rearrange(value, "b h w c -> b c h w").contiguous().float()

    def get_first_stage_encoding(self, encoded):
        latent = encoded if isinstance(encoded, torch.Tensor) else encoded.sample()
        return self.scale_factor * latent

    def encode_first_stage(self, value):
        return self.first_stage_model.encode(value)

    def decode_first_stage(self, value):
        return self.first_stage_model.decode(value / self.scale_factor)

    def encode_condition(self, value):
        encoded = self.encode_first_stage(value)
        latent = encoded if isinstance(encoded, torch.Tensor) else encoded.mode()
        return self.scale_factor * latent

    def encode_hdr_stage(self, value):
        encoded = self.hdr_stage_model.encode(value)
        latent = encoded if isinstance(encoded, torch.Tensor) else encoded.mode()
        return self.hdr_scale_factor * latent

    def decode_hdr_stage(self, value):
        return self.hdr_stage_model.decode(value / self.hdr_scale_factor)

    def get_condition(self, batch, count=None):
        perspective = batch[self.perspective_key]
        known_mask = batch[self.mask_key]
        asg = batch[self.asg_key]
        sg = batch[self.sg_key]
        shifts = batch.get("sg_shift")
        flips = batch.get("sg_flip")
        if count is not None:
            perspective = perspective[:count]
            known_mask = known_mask[:count]
            asg = asg[:count]
            sg = sg[:count]
            shifts = shifts[:count] if shifts is not None else None
            flips = flips[:count] if flips is not None else None

        perspective = (
            rearrange(perspective, "b h w c -> b c h w").to(self.device).float()
        )
        known_mask = rearrange(known_mask, "b h w c -> b c h w").to(self.device).float()
        asg = rearrange(asg, "b h w c -> b c h w").to(self.device).float()
        sg = sg.to(self.device).float()

        perspective_z = self.encode_condition(perspective)
        asg_z = self.encode_condition(asg)
        known_mask = F.interpolate(
            known_mask,
            size=perspective_z.shape[-2:],
            mode="nearest",
        )
        known_mask = 1 - 2 * known_mask
        hdr = (
            stage2_sg_from_params(
                sg,
                perspective.shape[-1],
                perspective.shape[-2],
                shifts,
                flips,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        hdr_z = self.encode_hdr_stage(hdr)
        concat = torch.cat((perspective_z, known_mask, asg_z), dim=1)
        return {"c_concat": [concat], "c_hdr": [hdr_z]}

    @torch.no_grad()
    def get_input(self, batch, count=None, return_reconstruction=False):
        value = self.get_input_tensor(batch)
        if count is not None:
            value = value[:count]
        value = value.to(
            self.device,
            non_blocking=self.non_blocking_transfer,
        )
        latent = self.get_first_stage_encoding(self.encode_first_stage(value)).detach()
        condition = self.get_condition(batch, count)
        if return_reconstruction:
            return latent, condition, value, self.decode_first_stage(latent)
        return latent, condition

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.non_blocking_transfer:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        return {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def q_sample(self, value, timesteps, noise):
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, value.shape) * value
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, value.shape)
            * noise
        )

    def apply_model(self, value, timesteps, condition):
        if self.ema_enabled and not self.training and self.ema_weights:
            return functional_call(
                self.model,
                self.ema_weights,
                (value, timesteps, condition),
                strict=False,
            )
        return self.model(value, timesteps, condition)

    def prediction_loss(self, prediction, target, reduction="mean"):
        return F.mse_loss(prediction, target, reduction=reduction)

    def forward(self, value, condition):
        timesteps = torch.randint(
            0,
            self.num_timesteps,
            (value.shape[0],),
            device=self.device,
        )
        noise = torch.randn_like(value)
        predicted_noise = self.apply_model(
            self.q_sample(value, timesteps, noise),
            timesteps,
            condition,
        )
        loss_values = self.prediction_loss(
            predicted_noise,
            noise,
            reduction="none",
        )
        loss_per_item = loss_values.mean(dim=(1, 2, 3))
        loss = loss_per_item.mean()
        level_loss = (self.lvlb_weights[timesteps] * loss_per_item).mean()
        prefix = "train" if self.training else "val"
        return loss, {
            f"{prefix}/loss_simple": loss,
            f"{prefix}/loss_vlb": level_loss,
            f"{prefix}/loss": loss,
        }

    def shared_step(self, batch):
        return self(*self.get_input(batch))

    def training_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch)
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch[self.first_stage_key].shape[0],
        )
        self.log("global_step", self.global_step, prog_bar=True, on_step=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, metrics = self.shared_step(batch)
        self.log_dict(
            metrics,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch[self.first_stage_key].shape[0],
        )

    def lr_multiplier(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / self.warmup_steps
            value = self.warmup_start + progress * (
                self.learning_rate - self.warmup_start
            )
            return value / self.learning_rate
        decay_start = self.warmup_steps + self.constant_steps
        if step < decay_start:
            return 1.0
        if self.decay_steps > 0:
            decay_step = max(step - decay_start, 0)
            progress = min(decay_step / self.decay_steps, 1.0)
            weight = 0.5 * (1 + cos(pi * progress))
            value = self.decay_target + weight * (
                self.learning_rate - self.decay_target
            )
            return value / self.learning_rate
        return 1.0

    def configure_optimizers(self):
        optimizer_options = {}
        if self.fused_optimizer and torch.cuda.is_available():
            optimizer_options["fused"] = True
        learning_rates = {
            "base": self.learning_rate,
            "condition": self.learning_rate * self.cond_lr_scale,
            "hdr": self.learning_rate * self.hdr_lr_scale,
        }
        parameters = []
        for name, group_parameters in self.diffusion_parameter_groups().items():
            trainable = [
                parameter for parameter in group_parameters if parameter.requires_grad
            ]
            if trainable:
                parameters.append(
                    {
                        "name": name,
                        "params": trainable,
                        "lr": learning_rates[name],
                    }
                )
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            **optimizer_options,
        )
        if self.warmup_steps <= 0 and self.decay_steps <= 0:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, self.lr_multiplier),
                "interval": "step",
                "frequency": 1,
            },
        }

    def sample_noise(self, batch_size, shape, generator=None, seeds=None):
        if seeds is None:
            return torch.randn(
                (batch_size, *shape),
                device=self.device,
                generator=generator,
            )
        if len(seeds) != batch_size:
            raise ValueError(f"expected {batch_size} sample seeds, got {len(seeds)}")
        return torch.cat(
            [
                torch.randn(
                    (1, *shape),
                    device=self.device,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                )
                for seed in seeds
            ]
        )

    @torch.no_grad()
    def sample_batch(
        self,
        batch,
        ddim_steps=50,
        eta=0,
        generator=None,
        seeds=None,
    ):
        condition = self.get_condition(batch)
        concat = condition["c_concat"][0]
        batch_size = concat.shape[0]
        shape = (self.channels, concat.shape[-2], concat.shape[-1])

        noise = self.sample_noise(batch_size, shape, generator, seeds)
        samples = DDIMSampler(self).sample(
            ddim_steps,
            batch_size,
            shape,
            condition,
            eta=eta,
            x_T=noise,
            verbose=False,
        )
        return self.decode_first_stage(samples)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        ddim_steps=50,
        seed=None,
        seeds=None,
        **kwargs,
    ):
        count_key = (
            self.first_stage_key
            if self.first_stage_key in batch
            else self.perspective_key
        )
        count = min(N, batch[count_key].shape[0])
        log_batch = {
            key: value[:count] if isinstance(value, (torch.Tensor, list)) else value
            for key, value in batch.items()
        }
        if seeds is not None:
            seeds = seeds[:count]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        samples = self.sample_batch(
            log_batch,
            ddim_steps=ddim_steps,
            generator=generator,
            seeds=seeds,
        )
        perspective = (
            rearrange(
                log_batch[self.perspective_key],
                "b h w c -> b c h w",
            )
            .to(self.device)
            .float()
        )
        asg = (
            rearrange(log_batch[self.asg_key], "b h w c -> b c h w")
            .to(self.device)
            .float()
        )
        sg = log_batch[self.sg_key].to(self.device).float()
        shifts = log_batch.get("sg_shift")
        flips = log_batch.get("sg_flip")
        hdr = (
            stage2_sg_from_params(
                sg,
                perspective.shape[-1],
                perspective.shape[-2],
                shifts,
                flips,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        hdr_reconstruction = self.decode_hdr_stage(self.encode_hdr_stage(hdr))
        hdr_reconstruction = repeat(
            hdr_reconstruction,
            "b 1 h w -> b c h w",
            c=3,
        )
        views = [perspective, asg, hdr_reconstruction]
        if self.first_stage_key in log_batch:
            _, _, _, reconstruction = self.get_input(
                log_batch,
                return_reconstruction=True,
            )
            views.append(reconstruction)
        views.append(samples)
        return {
            f"comparison_{index:02d}": torch.cat(
                [view[index : index + 1] for view in views],
                dim=-1,
            )
            for index in range(count)
        }


def create_model(config_path):
    config = load_config(config_path)
    return PanoVAEDiffusion(
        config["diffusion"],
        config["unet"],
        config["autoencoder"],
        config["hdr_autoencoder"],
    )
