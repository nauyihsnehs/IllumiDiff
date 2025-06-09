import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pathlib
import random
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict

import cv2 as cv

exr_save_params = [48, 1, 49, 4]


def get_mask():
    mask = cv.imread('inpainting-mask.png')
    mask = cv.resize(mask, (64, 32), interpolation=cv.INTER_NEAREST)
    mask = np.sum(mask, axis=-1, keepdims=True)
    mask[mask < 30] = 0
    mask[mask >= 30] = 1
    return mask


def run_sampler(
        model,
        input_image: np.ndarray,
        hint,
        hdr,
        prompt,
        img_name,
        batch: int = 10,
        num_samples=1,
        seed: int = -1,
        guess_mode=False,
        strength=1.0,
        ddim_steps=20,
        eta=0.0,
        show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        ddim_sampler = DDIMSampler(model)

        mask = get_mask()

        H, W, C = input_image.shape
        x0 = torch.from_numpy(input_image).clone().float().cuda()  # / 127.5 - 1.0
        x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
        x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

        control = torch.from_numpy(hint).clone().float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        control_hdr = torch.from_numpy(hdr).clone().float().cuda()
        control_hdr = torch.stack([control_hdr for _ in range(num_samples)], dim=0)
        control_hdr = einops.rearrange(control_hdr, 'b h w c -> b c h w').clone()

        prompt = torch.stack([prompt for _ in range(batch)], dim=0)

        mask = mask.astype('uint8')
        mask = torch.from_numpy(mask).clone().float().cuda()
        mask = torch.stack([mask for _ in range(num_samples)], dim=0)
        mask = einops.rearrange(mask, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {
            "c_concat": [control],
            "c_hdr": [control_hdr],
            "c_crossattn": [model.get_learned_conditioning(prompt)],
        }

        shape = (4, H // 8, W // 8)
        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)

        x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13))

        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            batch_size=batch,
            shape=shape,
            conditioning=cond,
            verbose=False,
            eta=eta,
            x0=x0,
            mask=mask,
            show_progress=show_progress,
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')).cpu().numpy().astype(np.float32)

        for id, x_sample in enumerate(x_samples):
            x_sample = (x_sample + 1) / 2
            x_sample = x_sample[..., ::-1]
            ldr = np.clip(x_sample * 255, 0, 255)
            cv.imwrite(f'inpainting_data/output/{img_name}_{str(id).zfill(2)}.jpg', ldr)


if __name__ == '__main__':
    cmodel = create_model('./models/cldm_v15_clip_hdr_share.yaml').cpu()
    model_path = 'epoch=19-step=191800.ckpt'
    model_name = model_path.split('-')[0]
    cmodel.load_state_dict(load_state_dict(f'lightning_logs/version_6/checkpoints/{model_path}', location='cuda'))
    pano_paths = [p.as_posix() for p in pathlib.Path('inpainting_data/input/').glob('*_pano.jpg')]
    asg_paths = [p.as_posix() for p in pathlib.Path('inpainting_data/input/').glob('*_asg.jpg')]
    img_paths = [p.as_posix() for p in pathlib.Path('inpainting_data/input/').glob('*_input.png')]
    sg_paths = [p.as_posix() for p in pathlib.Path('inpainting_data/input/').glob('*_sg.exr')]

    for path in tqdm(pano_paths):
        in_img_name = path.split('/')[-1].split('-')[0]
        save_name = path.split('/')[-1].split('.')[0] + f'_{model_name}'
        in_pano = cv.imread(path)
        in_pano = cv.resize(in_pano, (512, 256))
        in_pano = cv.cvtColor(in_pano, cv.COLOR_BGR2RGB)
        in_pano = (in_pano / 255.).astype(np.float32)
        in_pano = (in_pano - in_pano.min()) / (in_pano.max() - in_pano.min())
        in_pano = in_pano * 2 - 1

        in_asg_path = None
        for asg_path in asg_paths:
            if asg_path.split('/')[-1].startswith(in_img_name):
                in_asg_path = asg_path
                break
        if in_asg_path is None: exit(0)
        in_asg = cv.imread(in_asg_path)
        in_asg = cv.resize(in_asg, (512, 256))
        in_asg = cv.cvtColor(in_asg, cv.COLOR_BGR2RGB)
        in_asg = (in_asg / 255.).astype(np.float32)
        in_asg = (in_asg - in_asg.min()) / (in_asg.max() - in_asg.min())
        in_asg = in_asg * 2 - 1

        in_sg_path = None
        lum_weight = np.asarray([0.2126, 0.7152, 0.0722])[None, None, ...]
        for sg_path in sg_paths:
            if sg_path.split('/')[-1].startswith(in_img_name):
                in_sg_path = sg_path
                break
        if in_sg_path is None: exit(0)
        in_sg = cv.imread(in_sg_path, cv.IMREAD_UNCHANGED)
        in_sg = cv.resize(in_sg, (512, 256))
        # in_sg = cv.cvtColor(in_sg, cv.COLOR_BGR2RGB)
        in_sg = np.sum(in_sg * lum_weight, axis=-1, keepdims=True)
        in_sg = np.log10(in_sg + 1)
        in_sg = in_sg / in_sg.max() if in_sg.max() > 0.3 else in_sg
        in_sg = (in_sg - in_sg.min()) / (in_sg.max() - in_sg.min())
        in_sg = in_sg * 2 - 1

        in_img_path = None
        for img_path in img_paths:
            if img_path.split('/')[-1].startswith(in_img_name):
                in_img_path = img_path
                break
        if in_img_path is None: exit(0)
        in_img = cv.imread(in_img_path)
        in_img = cv.resize(in_img, (224, 224))
        in_img = cv.cvtColor(in_img, cv.COLOR_BGR2RGB) / 255
        in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())
        in_img = torch.from_numpy(in_img).permute(2, 0, 1).float().cuda()

        run_sampler(cmodel, in_pano, in_asg, in_sg, in_img, save_name, seed=3407)
