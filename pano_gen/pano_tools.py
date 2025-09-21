import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pathlib
from tqdm import tqdm
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

# exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]
exr_save_params = [48, 1, 49, 4]


def interpolate(pano, u, v, valid=None, order=1, prefilter=True):
    h, w, d = pano.shape
    coords = [(v * h).ravel(), (u * w).ravel()]
    out = np.stack([map_coordinates(pano[..., c], coords, order=order, mode='nearest', prefilter=prefilter)
                   .reshape(u.shape) for c in range(pano.shape[2])], axis=-1)
    if valid is not None and np.invert(valid).sum() > 0:
        mask = np.invert(valid)
        # out[np.tile(mask[:, :, None], (1, 1, d))] = np.tile(np.zeros(d), (mask.sum()))
        out[mask] = 0
    return out


def image2world(u, v):
    theta = np.pi * (u * 2 - 1)
    phi = np.pi * v
    x = np.sin(phi) * np.sin(theta)
    y = np.cos(phi)
    z = -np.sin(phi) * np.cos(theta)
    return x, y, z, np.ones_like(x, bool)


def world2image(x, y, z):
    u = 0.5 * (1 + np.arctan2(x, -z) / np.pi)
    v = np.arccos(y) / np.pi
    return u, v


def world_coordinates(w, h, dtype='float32'):
    cols = np.linspace(0, 1, w * 2 + 1)[1::2]  # pano.shape[1]
    rows = np.linspace(0, 1, h * 2 + 1)[1::2]  # pano.shape[0]
    u, v = [d.astype(dtype) for d in np.meshgrid(cols, rows)]
    return image2world(u, v)


def rotate(pano, el, az, ro=0, order=1):
    dcm = R.from_euler('ZXY', [ro, el, -az]).as_matrix()
    dx, dy, dz, valid = world_coordinates(pano.shape[1], pano.shape[0])
    ptR = dcm @ np.vstack((dx.ravel(), dy.ravel(), dz.ravel()))
    dx, dy, dz = [np.clip(a.reshape(dx.shape), -1, 1) for a in ptR]
    u, v = world2image(dx, dy, dz)
    return interpolate(pano, u, v, valid=valid, order=order)


def camera_coordinates(pano, vfov, rotation_ear, ar=4. / 3., resolution=(640, 480)):
    hfov = 2 * np.arctan(np.tan(vfov * np.pi / 180. / 2) * ar) * 180 / np.pi
    mu = np.tan(hfov / 2. * np.pi / 180.)
    mv = np.tan(vfov / 2. * np.pi / 180.)
    x, y, z, _ = world_coordinates(pano.shape[1], pano.shape[0], dtype='float64')
    xy = np.sqrt((x ** 2 + y ** 2) / np.maximum(-x ** 2 - y ** 2 + 1, 1e-10))
    theta = np.arctan2(x, y)
    x = xy * np.sin(theta)
    y = xy * np.cos(theta)
    hmask = (x > -mu) & (x < mu)
    vmask = (y > -mv) & (y < mv)
    dmask = z < 0
    mask = hmask & vmask & dmask
    mask = mask[..., None].astype('float64')
    mask = rotate(mask, *rotation_ear)[..., 0]
    dy = np.linspace(mv, -mv, resolution[1])
    dx = np.linspace(-mu, mu, resolution[0])
    x, y = np.meshgrid(dx, dy)
    x, y = x.ravel(), y.ravel()
    xy = np.sqrt((x ** 2 + y ** 2) / (x ** 2 + y ** 2 + 1))
    theta = np.arctan2(x, y)
    x = xy * np.sin(theta)
    y = xy * np.cos(theta)
    z = -np.sqrt(1 - (x ** 2 + y ** 2))
    coords = np.vstack((x, y, z))
    el, az, ro = rotation_ear
    coords = R.from_euler('ZXY', [ro, el, -az]).as_matrix().T.dot(coords)
    return mask, coords


def pano2pers(pano, vfov, rotation_ear, ar=1., resolution=(640, 480), order=1):
    mask, coords = camera_coordinates(pano, vfov, rotation_ear, ar, resolution)
    u, v = world2image(coords[0, :], coords[1, :], coords[2, :])
    u, v = u.reshape(resolution[::-1]), v.reshape(resolution[::-1])
    crop = interpolate(pano, u, v, order=order)
    return crop, mask


def pers2pano(pers, pano_shape, vfov, rotation_ear, order=1):
    W_pan, H_pan = pano_shape
    H_pers, W_pers, C = pers.shape
    ar = max(H_pers, W_pers) / min(H_pers, W_pers)
    mask, coords = camera_coordinates(np.zeros((H_pan, W_pan, C)), vfov, [0, 0, 0], ar, resolution=(W_pers, H_pers))
    u_pan, v_pan = world2image(coords[0, :], coords[1, :], coords[2, :])
    u_pan = u_pan.reshape((H_pers, W_pers))
    v_pan = v_pan.reshape((H_pers, W_pers))
    j_idx = np.clip((u_pan * W_pan).astype(int), 0, W_pan - 1)
    i_idx = np.clip((v_pan * H_pan).astype(int), 0, H_pan - 1)
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    pers_flat = pers.reshape(-1, C)
    pano_out = np.zeros((H_pan, W_pan, C), dtype=pers.dtype)
    pano_out[i_flat, j_flat] = pers_flat
    pano_out = rotate(pano_out, *rotation_ear, order=order)
    return pano_out, mask


def warping(pano, nadir=0.7, order=1):
    h, w, _ = pano.shape
    zoff = np.sin(nadir)
    x, y, z, _ = world_coordinates(w, h, dtype='float64')
    t = -z * zoff + np.sqrt(zoff * zoff * (z * z - 1) + 1)
    u, v = world2image(x * t, y * t, z * t + zoff)
    return interpolate(pano, u, v, order=order)


if __name__ == '__main__':
    def pers2pano_test():
        pers = cv.imread('crop_test02.exr', cv.IMREAD_UNCHANGED)
        pano_shape = (1024, 2048, 3)
        vfov = 60
        rotation_ear = [np.pi / 8, np.pi / 6, 0]
        pano, _ = pers2pano(pers, pano_shape, vfov, rotation_ear)
        cv.imwrite('pers2pano_test.exr', pano.astype(np.float32), exr_save_params)
        exit(0)


    def pano2pers_test():
        pano = cv.imread('9C4A0022-6d8fe2e88e.exr', cv.IMREAD_UNCHANGED)
        vfov = 60
        rotation_ear = [np.pi / 8, np.pi / 6, 0]
        ar = 4. / 3.
        resolution = (512, 512)
        crop, mask = pano2pers(pano, vfov, rotation_ear, ar=ar, resolution=resolution)
        cv.imwrite('crop_test01.exr', crop.astype(np.float32), exr_save_params)
        cv.imwrite('crop_mask01.exr', mask.astype(np.float32), exr_save_params)


    def pano_rotate_test():
        pano = cv.imread('9C4A0022-6d8fe2e88e.exr', cv.IMREAD_UNCHANGED)
        elevation = np.pi / 8
        azimuth = np.pi / 6
        roll = 0
        rotated_pano = rotate(pano, el=elevation, az=azimuth, ro=roll)
        cv.imwrite('rotated_pano.exr', rotated_pano.astype(np.float32), exr_save_params)


    def warping_test():
        pano = cv.imread('9C4A0022-6d8fe2e88e.exr', cv.IMREAD_UNCHANGED)
        nadir = -0.7
        warped_pano = warping(pano, nadir=nadir)
        cv.imwrite(f'warped_pano_{nadir}.exr', warped_pano.astype(np.float32), exr_save_params)


    def fov_warping_test():
        pers = cv.imread('9C4A0003-e05009bcad_r0.exr', cv.IMREAD_UNCHANGED)
        pers = cv.resize(pers, (1024, 1024), interpolation=cv.INTER_AREA)
        # vfov = 60
        # nadir = -0.7
        pano_shape = (512, 1024, 3)
        for vfov in [60, 65, 70, 75, 80, 90]:
            for nadir in [0, -0.5, -0.6, -0.7, -0.8, -0.9]:
                print(f'Processing FOV: {vfov}')
                pano, _ = pers2pano(pers, pano_shape, vfov, [0, 0, 0])
                warped_pano = warping(pano, nadir=nadir)
                warped_pano = np.clip(warped_pano ** (1 / 2.2), 0, 1) * 255
                cv.imwrite(f'warp_fov_nadir_test/warped_pano_fov_{vfov}_nadir_{nadir}.jpg', warped_pano)


    def rotate_warping_test():
        pers = cv.imread('9C4A0003-e05009bcad_r0.exr', cv.IMREAD_UNCHANGED)
        pers = cv.resize(pers, (1024, 1024), interpolation=cv.INTER_AREA)
        vfov = 75
        nadir = -0.7
        pano_shape = (512, 1024, 3)
        for el in [-np.pi / 6, -np.pi / 8, 0, np.pi / 8, np.pi / 6]:
            for az in [-np.pi / 6, -np.pi / 8, 0, np.pi / 8, np.pi / 6]:
                pano, _ = pers2pano(pers, pano_shape, vfov, [el, az, 0])
                warped_pano = warping(pano, nadir=nadir)
                warped_pano = np.clip(warped_pano ** (1 / 2.2), 0, 1) * 255
                cv.imwrite(f'warp_rotate_test/warped_pano_el_{round(el, 3)}_az_{round(az, 3)}.jpg', warped_pano)


    # pers2pano_test()
    # pano2pers_test()
    # pano_rotate_test()
    # warping_test()
    rotate_warping_test()

