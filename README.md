# IllumiDiff: ~~Indoor~~ Illumination Estimation from a Single Image with Diffusion Model (TVCG 2025)

[Shiyuan Shen](https://nauyihsnehs.github.io/), [Zhongyun Bao](https://www.ahpu.edu.cn/jsjyxxgc/2024/0829/c5472a228006/page.htm), [Wenju Xu](https://xuwenju123.github.io/), [Chunxia Xiao](https://graphvision.whu.edu.cn/)

**[Paper](https://ieeexplore.ieee.org/document/10945728)** | **[PDF](https://graphvision.whu.edu.cn/paper/2025/ShenShiYuan_TVCG_2025.pdf)** | **[HomePage](https://graphvision.whu.edu.cn/)**

###
[Update] We trained a new model with both indoor and outdoor support.

## Structure

```text
IllumiDiff/
├── ckpts/                      # Downloaded checkpoints for inference
├── lighting_est/               # Stage 1 and Stage 3 models, datasets, and training
├── pano_ldm/                   # canonical Stage 2 latent diffusion implementation
├── datasets/                   # preprocessing, fitting, splitting, and inpainting tools
├── utils.py                    # shared image, logging, runtime, and panorama utilities
├── inference.py                # full Stage 1 + 2 + 3 pipeline
└── inference_single.py         # parameterized SG + ASG lighting
```

## Environment

```bash
conda create -n illumidiff python=3.11
conda activate illumidiff
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Checkpoints

Download the checkpoints from
[OneDrive](https://1drv.ms/f/s!AteITnyFLzOYj6x_vV0lu5uhoTVjJQ?e=YJViCX)
and place them in `./ckpts/`. The default inference commands expect:

```text
id_net-step020k.ckpt
sg_net-step080k.ckpt
asg_net-step100k.ckpt
hdr_net-step038k.ckpt
ldm-step015k.ckpt
```

## Inference

Full lighting estimation + panorama generation + inverse tonemapping inference:

```bash
python inference.py --input-path <path> --output-path <path>
```

Linear SG + ASG output:

```bash
python inference_single.py --input-path <path> --output-path <path>
```

## Dataset

Start with a folder of HDR panoramas:

```bash
# Original HDR panoramas -> paired rotated pano/perspective HDR and LDR data
python datasets/build_dataset.py --input-dir <hdr-folder> --dataset-root <dataset_root>

# Fit parameterized lighting targets
python datasets/sg_fitting.py --dataset-root <dataset_root>
python datasets/asg_fitting.py --dataset-root <dataset_root>

# Preview a split before moving complete modality groups. Indoor and outdoor
# lists are independent; provide either one or both.
python datasets/split_testset.py --dataset-root <dataset_root>
```

## Training

All networks are trained separately.

For id_net, sg_net, asg_net or hdr_net:

```bash
python -m lighting_est.train --config lighting_est/configs/<network>.toml
```

For LDM:

```bash
python -m pano_ldm.train_vae_1ch --config pano_ldm/configs/train_vae.toml
python -m pano_ldm.train --config pano_ldm/configs/train.toml --init-ckpt <init-path>
```

## Contact

For questions, please contact:
[syshen@whu.edu.cn](mailto:syshen@whu.edu.cn)

## Citation

```bibtex
@article{shen2025illumidiff,
  title={IllumiDiff: Indoor Illumination Estimation from a Single Image with Diffusion Model},
  author={Shen, Shiyuan and Bao, Zhongyun and Xu, Wenju and Xiao, Chunxia},
  journal={IEEE transactions on visualization and computer graphics},
  year={2025},
  publisher={IEEE}
}
```