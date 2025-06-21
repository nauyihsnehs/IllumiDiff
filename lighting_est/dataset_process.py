import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pathlib

from PIL import Image
from tqdm import tqdm
# from simple_lama_inpainting import SimpleLama

import numpy as np
# from diffusers import StableDiffusionInpaintPipeline
# from diffusers import AutoPipelineForInpainting

import cv2 as cv

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]
lum_weight = np.asarray([0.0722, 0.7152, 0.2126])[None, None, ...]  # bgr
lum_percentile = 95
tonemap = cv.createTonemapReinhard(1.2, 1.25, 0.1, 0.)


def lum_calib(hdr, save_folder, save_name):
    lum = np.sum(hdr * lum_weight, axis=-1)
    lum_scale = 0.8 / np.percentile(lum, lum_percentile)
    img_calib = (hdr * lum_scale[..., None]).astype(np.float32)
    cv.imwrite(f'{save_folder}/{save_name}_{round(lum_scale, 8)}.exr', img_calib, exr_save_params)
    preview_img = cv.resize(img_calib, (1024, 512), interpolation=cv.INTER_AREA)
    preview_img = (np.clip(preview_img, 0, 1) * 255).astype(np.uint8)
    cv.imwrite(f'{save_folder}/preview/{save_name}_{round(lum_scale, 8)}.jpg', preview_img)


def get_light_source_mask(hdr, save_folder, save_name, is_save=True, is_flip=True):
    hdr = np.sum(hdr * lum_weight, axis=-1)
    threshold = min(np.percentile(hdr, 98), 2)
    mask = np.zeros_like(hdr)
    mask[hdr > threshold] = 1
    if is_save:
        cv.imwrite(f'{save_folder}/{save_name}.png', mask * 255)
        if is_flip: cv.imwrite(f'{save_folder}/{save_name}_flip.png', cv.flip(mask, 1) * 255)
    return mask * 255


def simple_inpaint_bottom_of_image(image, save_folder, save_name, k=1):
    height, width, channels = image.shape
    for x in range(width):
        for y in range(height // 3 * 2, height):
            if np.all(image[y, x] == [0, 0, 0]):
                non_zero_pixels = []
                for y_offset in range(1, k + 1):
                    if y - y_offset >= 0 and np.all(image[y - y_offset, x] != [0, 0, 0]):
                        non_zero_pixels.append(image[y - y_offset, x])
                if len(non_zero_pixels) > 0:
                    avg_color = np.mean(non_zero_pixels, axis=0)
                    image[y, x] = avg_color
    cv.imwrite(f'{save_folder}/{save_name}{ext}', image, exr_save_params)


def get_bottom_mask(hdr, save_folder, save_name):
    hdr = np.sum(hdr, axis=-1)
    mask = np.zeros_like(hdr)
    mask[hdr > 0] = 255
    cv.imwrite(f'{save_folder}/{save_name}.png', mask)


def get_ldr_clip(hdr, save_folder, save_name):
    ori_scale = float(save_name.split('_')[-1])
    save_name = save_name.split('_')[0]
    lum = np.sum(hdr * lum_weight, axis=-1)
    lum_scale = 0.8 / np.percentile(lum, 90)
    hdr_bright = hdr * lum_scale[..., None]
    scale = ori_scale * lum_scale
    ldr_clip = np.clip(hdr_bright, 0, 1) * 255
    cv.imwrite(f'{save_folder}/{save_name}_{round(scale, 8)}.png', ldr_clip)


def sd_inpainting(pipe, ldr, mask, save_folder, save_name):
    if mask is None:
        mask = np.ones_like(ldr) * 255
        mask[ldr == 0] = 0
    wo, ho = ldr.shape[:2]
    w, h = 1024, 1024
    ldr = cv.resize(ldr, (w, h))
    ldr = cv.cvtColor(ldr, cv.COLOR_BGR2RGB)
    ldr = Image.fromarray(ldr)
    mask = 255 - mask
    mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)
    mask = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = Image.fromarray(mask).convert('L')
    prompt = "down view of the clean floor in a room, clean, smooth, pure"
    negative_prompt = "text, object, human"
    image = pipe(width=w, height=h, prompt=prompt, negative_prompt=negative_prompt, image=ldr, mask_image=mask,
                 num_inference_steps=20).images[0]  # , strength=0.99, guidance_scale=8.0
    image = image.resize((wo, ho))
    image.save(f'{save_folder}/{save_name}.png')


def lama_inpainting(lama, ldr, mask, save_folder, save_name):
    ldr = cv.cvtColor(ldr, cv.COLOR_BGR2RGB)
    ldr = Image.fromarray(ldr)
    mask = 255 - mask
    mask = cv.GaussianBlur(mask, (9, 9), 6)
    mask = Image.fromarray(mask).convert('L')

    result = lama(ldr, mask)
    result.save(f'{save_folder}/{save_name}.png')


def pano_resize(pano, resolution, save_folder, save_name, ext, pers=False, is_save=True, is_flip=True):
    res = (resolution, resolution) if pers else (resolution, resolution // 2)
    pano = cv.resize(pano, res, interpolation=cv.INTER_AREA)
    if is_save:
        cv.imwrite(f'{save_folder}/{save_name}{ext}', pano, exr_save_params)
        if is_flip: cv.imwrite(f'{save_folder}/{save_name}_flip{ext}', cv.flip(pano, 1), exr_save_params)
    return pano


def pano_max_pooling(pano, save_folder, save_name, is_save=True):
    # max_pooling
    H, W = pano.shape[:2]
    h, w = 256, 512

    dh = H // h
    dw = W // w

    pad_h = abs(dh * h - H)
    pad_w = abs(dw * w - W)

    img_padded = np.pad(pano, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    weights = np.array([0.0722, 0.7152, 0.2126])
    brightness = np.dot(img_padded[..., :3], weights)

    pooled = np.zeros((h, w, pano.shape[2]), dtype=pano.dtype)

    for i in range(h):
        for j in range(w):
            y_start, y_end = i * dh, (i + 1) * dh
            x_start, x_end = j * dw, (j + 1) * dw

            region_brightness = brightness[y_start:y_end, x_start:x_end]
            region_img = img_padded[y_start:y_end, x_start:x_end]

            mask = region_brightness > 1
            region_brightness_filtered = region_brightness[mask]
            region_img_filtered = region_img[mask]

            if region_brightness_filtered.size > 0:
                pooled[i, j] = np.mean(region_img_filtered, axis=0)
            else:
                max_idx = np.unravel_index(np.argmax(region_brightness), (dh, dw))
                pooled[i, j] = region_img[max_idx[0], max_idx[1]]

    if is_save: cv.imwrite(f'{save_folder}/{save_name}.exr', pooled, exr_save_params)
    return pooled


def pano_rotate(img, save_folder, save_name, rotate_times=10, dsize=(512, 256), ext='.png', orient=1, is_flip=False):
    img = cv.resize(img, dsize)
    for i in range(rotate_times):
        rotate_distance = i * (img.shape[1] / rotate_times) * orient
        right_rotate = np.float32([[1, 0, rotate_distance], [0, 1, 0]])
        left_rotate = np.float32([[1, 0, rotate_distance - dsize[0] * orient], [0, 1, 0]])
        right_result = cv.warpAffine(img, right_rotate, dsize=dsize)
        left_result = cv.warpAffine(img, left_rotate, dsize=dsize)
        result = right_result + left_result
        cv.imwrite(f'{save_folder}/{save_name}_r{i}{ext}', result, exr_save_params)
        if is_flip: cv.imwrite(f'{save_folder}/{save_name}_r{i}_flip{ext}', cv.flip(result, 1), exr_save_params)


def pano_flip(img, save_folder, save_name, ext='.png'):
    flip = cv.flip(img, 1)
    cv.imwrite(f'{save_folder}/{save_name}_flip{ext}', flip, exr_save_params)


def multi_pano_pooling(i):
    ext = '.exr'
    pano_folder = ''
    pano_save_folder = ''
    pano_paths = [p.as_posix() for p in pathlib.Path(pano_folder).glob(f'*{ext}')]
    pano_paths.sort()
    img_path = pano_paths[i]
    save_name = img_path.replace('\\', '/').split('/')[-1].split(ext)[0].split('_')[0]
    if not os.path.exists(f'{pano_save_folder}/{save_name}_r{i}{ext}'):
        pano = cv.imread(img_path) if ext == '.png' else cv.imread(img_path, cv.IMREAD_UNCHANGED)
        pano_max_pooling(pano, pano_save_folder, save_name)


if __name__ == '__main__':
    img_folder = r'E:\Laval\laval_repair\laval_repair_hdr_1024'
    save_folder = r'E:\Laval\laval_repair\laval_repair_hdr_512_rotate_filp'
    ext = '.exr'
    img_paths = [p.as_posix() for p in pathlib.Path(img_folder).glob(f'*.*')]

    for img_path in tqdm(img_paths):
        img_name = img_path.replace('\\', '/').split('/')[-1].split(f'{ext}')[0]
        mask_name = img_name.split('_')[0]
        if os.path.exists(f'{save_folder}/input_ls_256/{img_name}.png'):
            continue
        if 'exr' in ext or 'hdr' in ext:
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        else:
            img = cv.imread(img_path)
        pano_rotate(img, save_folder, img_name, 10, dsize=(512, 256), ext=ext, orient=1, is_flip=True)
