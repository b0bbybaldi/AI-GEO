#!/usr/bin/env python3
import numpy as np

def fix_pixels(image, bad_pixels):
    bad_indices = np.nonzero(bad_pixels)
    mean = np.zeros(bad_indices[0].size, dtype=np.float32)
    mean_good = np.zeros(bad_indices[0].size, dtype=np.float32)
    norm = np.zeros(bad_indices[0].size, dtype=np.int32)
    shifts = ((-1,-1),
              (-1, 0),
              (-1, 1),
              ( 0, 1),
              ( 1, 1),
              ( 1, 0),
              ( 1,-1),
              ( 0,-1))

    for di, dj in shifts:
        I = np.clip(bad_indices[0] + di, 0, image.shape[0]-1)
        J = np.clip(bad_indices[1] + dj, 0, image.shape[1]-1)
        px = image[I, J]
        bmask = bad_pixels[I, J]
        mean += px
        mean_good += np.where(bmask, 0, px)
        norm += (~bmask).astype(np.int32)
    mean /= len(shifts)
    mean_good /= np.maximum(1, norm)
    image[bad_indices[0], bad_indices[1]] = np.where(norm <= len(shifts)/2, mean, mean_good)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from medpy.filter.smoothing import anisotropic_diffusion
    from scipy.ndimage.filters import gaussian_filter
    from map_envi import load
    
    #imt = np.load('tnsr.npy')[3000:4000,4000:5000,0]
    #bdt = np.load('bad_data.npy')[3000:4000,4000:5000]
    #print(imt.min(), imt.max(), imt.mean(), imt.std())

    # im = np.zeros((1000, 1000), dtype=np.float32)
    # bd = np.zeros((1000, 1000), dtype=np.bool)
    # for fn in ('Sigma0_IW1_VH_mst_22Jun2018',
    #            'Sigma0_IW1_VH_slv1_04Jul2018',
    #            'Sigma0_IW1_VV_mst_22Jun2018',
    #            'Sigma0_IW1_VV_slv2_04Jul2018'):
    #     patch, _ = load(fn)
    #     patch = patch[10000:11000,11000:12000].astype(np.float32)
    #     bd |= (patch <= 1e-6) | (patch > 10)
    #     im += patch

    # im *= 0.25
    #im = np.where(bd, 0, np.log10(np.maximum(1e-6, im)) )

    im1 = load('Sigma0_IW1_VV_mst_22Jun2018')[0][7000:13000,7000:13000].astype(np.float32)
    im2 = load('Sigma0_IW1_VH_mst_22Jun2018')[0][7000:13000,7000:13000].astype(np.float32)

    im = np.hypot(im1, im2)
    bd = (im <= 1e-6) | (im > 10)
    
    im = np.arcsin(im2 / im)
    
    im = np.maximum(1e-6, im)
    imor = im.copy()

    fix_pixels(im, bd)

    u = gaussian_filter(im, 4)
    u = anisotropic_diffusion(im, 50, 20, 0.25, option=1)
    #u = im
    G1 = np.hypot(*np.gradient(im))
    G2 = np.hypot(*np.gradient(u))

    plt.subplot(2, 2, 1), plt.imshow(imor, cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(u , cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(G1, cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(G2, cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(bottom=0., right=1, top=1)

    plt.show()
