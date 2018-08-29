#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcmap
from scipy.ndimage.filters import gaussian_filter, median_filter

pred = np.load('prob_pred_full.npy')

print(pred.shape)
#print(pred.max())
#print(pred.min())
print(pred.dtype)



if False:
    vis = np.where(pred.max(axis=-1) == 0, 0 , np.argmax(pred, axis=2)+1)
elif True:
    vis = np.zeros(pred.shape[:-1], dtype=np.uint8)

    water = ((pred[...,0] > 0.05)|(pred[...,6] > 0.05)).astype(np.bool)

    xz = (pred[...,4] > 0.1).astype(np.bool)
    build = (pred[...,3] > 0.1).astype(np.bool)
    manmade = (pred[...,1] > 0.1).astype(np.bool)
    manmade_2 = (pred[...,5] > 0.1).astype(np.bool)

    green_1 = (pred[...,2] > 0.3).astype(np.bool)
    #green_2 = (pred[...,3] > 0.3).astype(np.bool)


    vis = np.where(green_1, 6, 0)
    vis = np.where(xz, 5, vis)
    vis = np.where(manmade_2, 4, vis)
    vis = np.where(manmade, 3, vis)
    vis = np.where(build, 2, vis)
    vis = np.where(water, 1, vis)
    
    vis = np.where(vis == 0, np.where(pred.max(axis=-1) == 0, 0 , np.argmax(pred, axis=2)+1) , vis)
else:
    vis = (pred[...,5] > 0.05).astype(np.bool)
    np.save('tovect.npy', vis)
print(np.unique(vis))

#vis = median_filter(vis,4)

cs3 = plt.get_cmap('Set3').colors
colors = []
colors.append((0,0,0))
#colors.extend(cs3)

colors.append((0,0,1.))

colors.append((1.,0.,0.))
colors.append((1.,0.3,0))
colors.append((1.,0.7,0))
colors.append((1.,0.99,0))

colors.append((0.,1.,0.3))
#colors.append((0.,1.,0.8))

#colors.append((1.,0.,1.))
#cmap.colors = tuple(colors)
cmap = lcmap(colors)

if True:
    import map_envi as envi
    import re
    print('clusterized')
    fnames = envi.locate()
    fnames = [el for el in fnames if not re.search(r'_[dD][bB]', el)]
    fnames = [el for el in fnames if re.search(r'(Sigma)|(coh)', el)]
    full_shape, hdict = envi.read_header(fnames[0])

    colors = (np.array(colors) * 254).astype(np.uint8)
    imgc = np.empty(vis.shape + (3,))
    np.choose(vis, colors[...,0], out=imgc[...,0])
    np.choose(vis, colors[...,1], out=imgc[...,1])
    np.choose(vis, colors[...,2], out=imgc[...,2])
    imgc[0,0,:] = np.array((0,0,0), dtype=np.uint8)
    imgc[-1,-1,:] = np.array((255,255,255), dtype=np.uint8)

    envi.save('pred8c', imgc.astype(np.uint8), hdict['map info'], hdict['coordinate system string'], chnames=['R', 'G', 'B'], desc='clusterization, colors')
    print('saved')

plt.imshow(vis, cmap=cmap)
plt.show()
