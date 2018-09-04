#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture as GM
from scipy.ndimage.filters import gaussian_filter
from pickle import dump, load
from sys import argv
from sklearn.utils import shuffle
from os import system
import json

if len(argv) == 3:
    if argv[1] == 'fit':
        mode = 'f'
    elif argv[1] == 'fitpredict':
        mode = 'fp'
    elif argv[1] == 'predict':
        mode = 'p'
    else:
        print('Usage: [fit|predict|fitpredict] [zone|full]')
        exit(-1)

    if argv[2] == 'zone':
        mode_size = 'z'
        tnsr = np.load('tnsr_zone.npy')
        bad_data = np.load('bd_zone.npy')
    elif argv[2] == 'full':
        mode_size = 'f'
        tnsr = np.load('tnsr_full.npy')
        bad_data = np.load('bd_full.npy')
    else:
        print('Usage: [fit|predict|fitpredict] [zone|full]')
        exit(-1)
else:
    print('Usage: [fit|predict|fitpredict] [zone|full]')
    exit(-1)

recipe = json.load(open('recipe.json', 'r'))
cselect = tuple(recipe['learn_channels'])

tnsr = tnsr[...,cselect]
print(tnsr.shape)
ts = tnsr.shape

if mode == 'fp' or mode == 'f':
    gauss_sz = recipe['learn_gauss']
    if mode == 'fp':
        tnsr_or = tnsr.copy()
    tnorm = np.empty((tnsr.shape[-1], 2))
    for n in range(tnsr.shape[-1]):
        tnorm[n,0] = tnsr[...,n].mean()
        tnsr[...,n] -= tnorm[n,0]
        tnorm[n,1] = tnsr[...,n].std()
        tnsr[...,n] /= tnorm[n,1]
        #tnsr[bad_data,n] = tnorm[n,0]
        if gauss_sz:
            tnsr[...,n] = gaussian_filter(tnsr[...,n], gauss_sz)

    tnsr_learn = tnsr[~bad_data, :].reshape((-1, tnsr.shape[-1]))
    #tnsr_learn = tnsr.reshape((-1, tnsr.shape[-1]))
    print(tnsr_learn.shape)
    tnsr_learn = shuffle(tnsr_learn)
    
    predictor = MiniBatchKMeans(n_clusters=7, batch_size=1000000, compute_labels=False).fit(tnsr_learn)

    dump(predictor, open('predictor.pkl','wb'))
    np.save('tnorm.npy', tnorm)
    
    cc = np.array(predictor.cluster_centers_)    
    print(cc)
    gm = GM(cc.shape[0], max_iter=10, means_init = cc, tol=0.01)
    gm.fit(shuffle(tnsr_learn)[:(4000000 if mode_size == 'f' else 2000000)])
    print('gm')
    dump(gm, open('gm.pkl','wb'))
    system("say 'learning done'")
    if mode == 'fp':
        tnsr = tnsr_or
    else:
        exit(0)


tnorm = np.load('tnorm.npy')
#predictor = load(open('predictor.pkl','rb'))
gm = load(open('gm.pkl','rb'))
Ncc = len(gm.weights_)
prob_pred = np.empty(tnsr.shape[:-1] + (Ncc,), dtype=np.float32)

gauss_sz = recipe['predict_gauss']
ns = 100
d=6
for i in range(0, tnsr.shape[0], ns):
    print(i)
    d1 = min(d,i)
    d2 = max(0, min(tnsr.shape[0] - i - ns, d))
    tstr = tnsr[i-d1:i+ns+d2,:,:].copy()
    bdstr = bad_data[i-d1:i+ns+d2,:]

    strshape = tstr.shape
    tstr -= tnorm[np.newaxis, np.newaxis, :,0]
    tstr /= tnorm[np.newaxis, np.newaxis, :,1]
    #tstr[bdstr, :] = tnorm[:,0]
    if gauss_sz:
        for n in range(tstr.shape[-1]):
            tstr[...,n] = gaussian_filter(tstr[...,n], gauss_sz)
    
    ppstr = gm.predict_proba(tstr.reshape((-1, strshape[-1])))
    ppstr = ppstr.astype(np.float32).reshape(strshape[:-1] + (Ncc,))
    #prob_pred[i:i+ns,...] = ppstr
    ppstr = np.where(bdstr[...,np.newaxis], 0, ppstr)
    if d2 ==0:
        prob_pred[i:i+ns,...] = ppstr[d1:,...]
    else:
        prob_pred[i:i+ns,...] = ppstr[d1:-d2,...]
                             
if mode_size == 'z':
    np.save('prob_pred.npy', prob_pred)
else:
    np.save('prob_pred_full.npy', prob_pred)
                             
system("say 'prediction done'")
