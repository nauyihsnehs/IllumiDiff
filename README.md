# IllumiDiff: Indoor Illumination Estimation from a Single Image with Diffusion Model (TVCG 2025)

[Shiyuan Shen](https://nauyihsnehs.github.io/), [Zhongyun Bao](https://www.ahpu.edu.cn/jsjyxxgc/2024/0829/c5472a228006/page.htm), [Wenju Xu](https://xuwenju123.github.io/), [Chunxia Xiao](https://graphvision.whu.edu.cn/)

**[Paper](https://ieeexplore.ieee.org/document/10945728)** |
**[PDF](https://graphvision.whu.edu.cn/paper/2025/ShenShiYuan_TVCG_2025.pdf)** |
**[HomePage](https://graphvision.whu.edu.cn/)**

#### Differences from the original paper

1. Upgrade LDM to Stable Diffusion 1.5, using input images instead of text prompts.
2. Replace Latent HDR Guidance with an equivalent substitution using ControlNet, with the aim of accelerating the fine-tuning process.
3. Training epochs are changed to 40 for id_net, 50 for sg_net, 100 for asg_net, 50 for hdr_net, 10 for controlnet.

## Structure

```
IllumiDiff/
├── ckpts/                        # Pre-trained model checkpoints
├── lighting\_est/                 # Stage 1 (id\_net, sg\_net, asg\_net) & Stage 3 (hdr\_net)
│   ├── asg\_fitting\_fixed\_ldr\_adam\_batch.py   # ASG ground truth fitting
│   ├── sg\_fitting\_free\_nadam.py              # SG ground truth fitting
│   ├── dataset.py                            # Dataset loader (Stage 1 & 3)
│   ├── dataset\_processing.py                 # Dataset preprocessing scripts
│   ├── models.py                             # Model definitions (Stage 1 & 3)
│   ├── modules.py                            # Lightning modules (Stage 1 & 3)
├── pano\_gen/                    # Stage 2: panorama generation
│   ├── cldm/                                 # ControlNet core
│   ├── configs/                              # Model configuration files
│   ├── ldm/                                  # Latent Diffusion Model core
│   ├── openai/                               # OpenAI CLIP model
│   ├── dataset.py                            # Dataset loader (Stage 2)
│   ├── pano\_tools.py                         # Panorama projection tools
│   ├── tool\_add\_control.py                   # Checkpoint initialization
│   ├── outpainting-mask.png                  # Outpainting mask
├── inference\_lighting\_est.py     # Inference script (Stage 1 or 3)
├── inference\_pano\_gen.py         # Inference script (Stage 2)
├── pipeline\_lighting\_est.py      # Lighting estimation pipeline (Stage 1 or 3)
├── pipeline\_full.py              # Full pipeline (Stage 1 & 2 & 3)
├── train\_lighting\_est.py         # Training script (Stage 1 or 3)
├── train\_pano\_gen.py             # Training script (Stage 2)
```

## TODO or not to do

- [ ] Config files for all networks.
- [ ] Simplify LDM code.
- [ ] Full dataset process script.
- [ ] Training all stages together.

## Environment

```bash
conda create -n illumidiff python=3.10
conda activate illumidiff
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=11.8 numpy=1.26.4 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install -r requirements.txt
```

## Checkpoints

You can download them from [OneDrive](https://1drv.ms/f/s!AteITnyFLzOYj6x_vV0lu5uhoTVjJQ?e=YJViCX).

Unzip `clip-vit-base-patch32.zip` to `IllumiDiff/pano_gen/openai/clip-vit-base-patch32/`,

put all ckpts into `IllumDiff/ckpts/`,

`control_sd15_clip_asg_sg.ckpt` is required only when training from scratch.

## Inference

Full pipeline inference, where the input is a single image:

```bash
python pipeline_full.py --input_path <path> --output_path <path>
```

Inference for Stage 1 or Stage 3 only:

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

For questions, please contact:
[syshen@whu.edu.cn](mailto:syshen@whu.edu.cn)

## References

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

