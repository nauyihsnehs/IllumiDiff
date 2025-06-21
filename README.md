# IllumiDiff: Indoor Illumination Estimation from a Single Image with Diffusion Model (TVCG 2025)

[Shiyuan Shen](https://nauyihsnehs.github.io/), [Zhongyun Bao](https://www.ahpu.edu.cn/jsjyxxgc/2024/0829/c5472a228006/page.htm), [Wenju Xu](https://xuwenju123.github.io/), [Chunxia Xiao](https://graphvision.whu.edu.cn/)

**[Paper](https://ieeexplore.ieee.org/document/10945728)** |
**[PDF](https://graphvision.whu.edu.cn/paper/2025/ShenShiYuan_TVCG_2025.pdf)** |
**[HomePage](https://graphvision.whu.edu.cn/)**

#### Differences from the original paper

1. Upgrade LDM to Stable Diffusion 1.5, use input image instead of text prompt.
2. Replace Latent HDR Guidance with an equivalent substitution using ControlNet, with the aim of accelerating the fine-tuning process.
3. Training epochs are changed to 40 for id_net, 50 for sg_net, 100 for asg_net, 50 for hdr_net, 4 for controlnet.

## Structure

```
IllumiDiff/
├── ckpts/ # Pre-trained model checkpoints
├── lighting_est/ # including stage1 (id_net,sg_net,asg_net) and stage3 (hdr_net)
│   ├── asg_fitting_fixed_ldr_adam_batch.py # fitting asg ground truth
│   ├── sg_fitting_free_nadam.py # fitting sg ground truth
│   ├── dataset.py # dataset for stage1 and stage3
│   ├── dataset_processing.py # some dataset processing scripts, still on organization
│   ├── models.py # model definitions for stage1 and stage3
│   ├── modules.py # lightning modules for stage1 and stage3
├── pano_gen/
│   ├── cldm # controlnet core codes
│   ├── configs # configuration files for model definition
│   ├── ldm # ldm core codes
│   ├── openai # CLIP model
│   ├── dataset.py # dataset for ldm
│   ├── pano_tools.py # some tools for panorama projection
│   ├── tool_add_control.py # ckpt initialization
│   ├── outpainting-mask.png # outpainting mask
├── inference_lighting_est.py # inference script for stage1 and stage3
├── inference_pano_gen.py # inference script for stage2
├── pipeline_lighting_est.py # pipeline for lighting estimation (stage1 + stage3)
├── pipeline_full.py # full pipeline for IllumiDiff (stage1 + stage2 + stage3)
├── train_lighting_est.py # training script for stage1 and stage3
├── train_pano_gen.py # training script for stage2
```

## TODO or not to do

- [ ] Config files for all networks.
- [ ] Simplify LDM code.
- [ ] Full dataset process script.
- [ ] Training all stage together.

## Environment

```bash
conda create -n illumidiff python=3.10
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=11.8 numpy=1.26.4 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install -r requirements.txt
```

## Checkpoints

You can download them from [OneDrive](https://1drv.ms/f/s!AteITnyFLzOYj6x_vV0lu5uhoTVjJQ?e=YJViCX).

Unzip `clip-vit-base-patch32.zip` to `IllumiDiff/pano_gen/openai/clip-vit-base-patch32/`,

put all ckpts to `IllumDiff/ckpts/`,

`control_sd15_clip_asg_sg.ckpt` is required solely for training from scratch.

## Inference

Full pipeline inference, the input is single images:

```bash
python pipeline_full.py --input_path <path> --output_path <path>
```

Stage 1 or Stage 3 only:

```bash
python pipeline_lighting_est.py --input_path <path> --input_pano_path <path> --output_path <path>
```

Single network only:

for id_net, sg_net, asg_net, or hdr_net:

```bash
python inference_lighting_est.py --task <network>
```

for pano_gen:

```bash
python inference_pano_gen.py
```

## Dataset

See more details in the paper.

## Training

All networks are trained separately.

For id_net, sg_net, asg_net or hdr_net:

```bash
python train_lighting_est.py --task <network>
```

For pano_gen:

```bash
python train_pano_gen.py --ckpt_path <path> --config_path <path>
```

## Contact

For questions please contact:  
[syshen@whu.edu.cn](mailto:syshen@whu.edu.cn)

## Renferences

[ControlNet](https://github.com/lllyasviel/ControlNet)

[LDM](https://github.com/CompVis/latent-diffusion)

[Skylibs](https://github.com/soravux/skylibs)

## Citation

```bibtex
@article{shen2025illumidiff,
  title={IllumiDiff: Indoor Illumination Estimation from a Single Image with Diffusion Model},
  author={Shen, Shiyuan and Bao, Zhongyun and Xu, Wenju and Xiao, Chunxia},
  journal={IEEE transactions on visualization and computer graphics},
  year={2025},
  publisher={IEEE}
}
