import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pathlib

import cv2 as cv
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from pano_gen.cldm.ddim_hacked import DDIMSampler
from pano_gen.cldm.model import create_model, load_state_dict
from pano_gen.pano_tools import pers2pano

exr_save_params = [48, 1, 49, 4]


def get_mask(mask, num_samples):
    mask = torch.stack([mask for _ in range(num_samples)], dim=0)
    mask = einops.rearrange(mask, 'b h w c -> b c h w')
    mask = F.interpolate(mask, scale_factor=1 / 8, mode='bilinear', align_corners=False)
    return mask


def run_sampler(
        model, input_pano, input_asg, input_sg, input_img,
        img_name, output_path, input_mask=None,
        batch: int = 4,  # batch size for sampling
        seed=-1,
        strength=1.0,
        ddim_steps=50,
        eta=0.3,
):
    with torch.no_grad():
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)
        mask = get_mask(input_mask, batch)

        H, W, C = input_pano.shape
        x0 = input_pano.clone()
        x0 = torch.stack([x0 for _ in range(batch)], dim=0)
        x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

        control = input_asg.clone()
        control = torch.stack([control for _ in range(batch)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        control_hdr = input_sg.clone()
        control_hdr = torch.stack([control_hdr for _ in range(batch)], dim=0)
        control_hdr = einops.rearrange(control_hdr, 'b h w c -> b c h w').clone()

        input_img = torch.stack([input_img for _ in range(batch)], dim=0)

        if seed >= 0: seed_everything(seed)

        cond = {
            "c_concat": [control],
            "c_hdr": [control_hdr],
            "c_crossattn": [model.get_learned_conditioning(input_img)],
        }

        shape = (4, H // 8, W // 8)
        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)

        x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))
        # x0_rand = torch.randn(batch, *shape).cuda()
        # x0 = x0 * mask + x0_rand * (1 - mask)

        model.control_scales = ([strength] * 13)

        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            batch_size=batch,
            shape=shape,
            conditioning=cond,
            verbose=False,
            eta=eta,
            x0=x0,
            mask=mask,
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')).cpu().numpy().astype(np.float32)

        for id, x_sample in enumerate(x_samples):
            x_sample = (x_sample + 1) / 2
            x_sample = x_sample[..., ::-1]
            ldr = np.clip(x_sample * 255, 0, 255)
            cv.imwrite(f'{output_path}/{img_name}_{str(id).zfill(2)}.jpg', ldr)


if __name__ == '__main__':
    cmodel = create_model('./pano_gen/configs/cldm_v15_clip_asg_sg.yaml').cpu()
    model_path = 'ckpts/control-epoch=4-step=47950.ckpt'
    epoch, step = map(int, __import__('re').search(r'epoch=(\d+)-step=(\d+)', model_path).groups())
    model_name = f'e{epoch:02d}_s{step:06d}'
    cmodel.load_state_dict(load_state_dict(model_path, location='cuda'))
    cmodel.eval()

    input_paths = sorted([p.as_posix() for p in pathlib.Path('pano_gen/inpainting_data/inputs').glob('*.*')])
    asg_paths = 'pano_gen/inpainting_data/asg'
    sg_paths = 'pano_gen/inpainting_data/sg'
    mask_path = 'pano_gen/outpainting-mask.png'

    pano_res = (512, 256)
    mask = np.array(Image.open(mask_path).convert("L").resize(pano_res, resample=0)) / 255.0
    in_mask = torch.from_numpy(mask).float().cuda()[..., None]
    lum_weight = np.asarray([0.2126, 0.7152, 0.0722])[None, None, ...]  # rgb

    save_path = f'pano_gen/inpainting_data/output/{model_name}'
    os.makedirs(save_path, exist_ok=True)


    def to_norm_tensor(ndarray):
        tensor = torch.from_numpy(ndarray).float().cuda()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return tensor * 2 - 1


    for input_path in tqdm(input_paths):
        in_img_name = input_path.split('/')[-1][:-4]
        save_name = input_path.split('/')[-1].split('.')[0] + f'_{model_name}'

        in_img = np.array(Image.open(input_path).convert("RGB")) / 255.0
        in_pano = pers2pano(in_img, pano_res, vfov=90, rotation_ear=[0, 0, 0])
        in_pano = to_norm_tensor(in_pano)
        in_img = cv.resize(in_img, (224, 224))
        in_img = to_norm_tensor(in_img).permute(2, 0, 1)

        in_asg_path = [p.as_posix() for p in pathlib.Path(asg_paths).glob(f'*{in_img_name}*')][0]
        in_asg = np.array(Image.open(in_asg_path).convert("RGB").resize(pano_res)) / 255.0
        in_asg = to_norm_tensor(in_asg)
        in_asg = torch.concat((in_asg, 1 - in_mask), dim=-1)

        in_sg_path = [p.as_posix() for p in pathlib.Path(sg_paths).glob(f'*{in_img_name}*')][0]
        in_sg = cv.imread(in_sg_path, cv.IMREAD_UNCHANGED)
        in_sg = cv.resize(cv.cvtColor(in_sg, cv.COLOR_BGR2RGB), pano_res)
        in_sg = np.sum(in_sg * lum_weight, axis=-1, keepdims=True)
        in_sg = torch.from_numpy(in_sg).float().cuda() - 1

        run_sampler(cmodel, in_pano, in_asg, in_sg, in_img, save_name, save_path, input_mask=in_mask, seed=3407)
