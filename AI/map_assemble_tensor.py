#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import map_envi as envi
import re
from sys import argv
from os import system
import json
from map_adfilter import fix_pixels
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage.filters import gaussian_filter

recipe = json.load(open('./recipe.json', 'r'))

sigmaAvgNames = recipe['channels'].get('sigma_avg', [])
sigmaNames = recipe['channels'].get('sigma', [])
sigmaVVNames = recipe['channels'].get('sigmaVV', [])
sigmaVHNames = recipe['channels'].get('sigmaVH', [])

cohAvgNames = recipe['channels'].get('coh_avg', [])
cohNames = recipe['channels'].get('coh', [])
cohVVNames = recipe['channels'].get('cohVV', [])
cohVHNames = recipe['channels'].get('cohVH', [])

channelNames = sigmaNames + sigmaAvgNames + sigmaVVNames + sigmaVHNames + cohNames + cohAvgNames + cohVVNames + cohVHNames

zone = np.array(recipe.get('zone'))
products = recipe['products']

full_shape, _ = envi.read_header(channelNames[0])
print('full shape:', full_shape)

if zone is not None:
    zone_shape = (zone[1][0]-zone[0][0], zone[1][1]-zone[0][1])
    print('Zone:\n', zone, '\nShape: ', zone_shape)

if len(argv) != 2:
    print("Usage: [zone|full|both]\n")
    exit(-1)
else:
    if argv[1] == 'zone':
        mode = 'zone'
        if zone is None:
            print('No zone info in recipe\n')
            exit(-1)
    elif argv[1] == 'full':
        mode = 'full'
    elif argv[1] == 'both':
        mode = 'both'
        if zone is None:
            print('No zone info in recipe\n')
            exit(-1)
    else:
        print('Usage: [zone|full|both]\n')
        exit(-1)

nproducts =  ((len(sigmaNames) if 'sigma' in products else 0) +
              (1 if 'sigma_avg' in products else 0) +
              (len(sigmaVVNames) if 'sigma_hypot' in products else 0) +
              (len(sigmaVVNames) if 'sigma_pol' in products else 0) +
              (len(cohNames) if 'coh' in products else 0) +
              (1 if 'coh_avg' in products else 0) +
              (len(cohVVNames) if 'coh_hypot' in products else 0) +
              (len(cohVVNames) if 'coh_pol' in products else 0)
)

if mode in ('zone', 'both'):
    tnsr_zone = np.empty((zone_shape[0], zone_shape[1], nproducts), dtype=np.float32)
    bd_zone = np.zeros((zone_shape[0], zone_shape[1]), dtype=np.bool)
if mode in ('full', 'both'):
    tnsr_full = np.empty((full_shape[0], full_shape[1], nproducts), dtype=np.float32)
    bd_full = np.zeros((full_shape[0], full_shape[1]), dtype=np.bool)

product_index = 0


if ('sigma' in products):
    params = products['sigma']
    for sn in sigmaNames:
        print(sn)
        s = envi.load(sn)[0]
        
        if mode == 'zone':
            s = s[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
        
        bad_data = (s < 1e-6) | (s > 10) | (s < 1e-6) | (s > 10)
        s = np.clip(s, 1e-6, 10)        
        s = np.log10(s)
        fix_pixels(s, bad_data)
        s = anisotropic_diffusion(s, params[0], params[1], 0.2, option=1)
        
        if mode == 'zone':
            tnsr_zone[..., product_index] = s
            product_index += 1
            bd_zone |= bad_data
        elif mode == 'full':
            tnsr_full[..., product_index] = s
            product_index += 1
            bd_full |= bad_data
        elif mode == 'both':
            tnsr_full[..., product_index] = s
            tnsr_zone[..., product_index] = s[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
            product_index += 1
            bd_full |= bad_data
            bd_zone |= bad_data[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]

if ('sigma_avg' in products):
    params = products['sigma_avg']
    if mode in ('zone', 'both'):
        savg_zone = np.zeros(zone_shape, dtype=np.float32)
    if mode in ('full', 'both'):
        savg_full = np.zeros(full_shape, dtype=np.float32)
        
    for sn in sigmaAvgNames:
        print(sn)
        s = envi.load(sn)[0]
        
        if mode == 'zone':
            s = s[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
        
        bad_data = (s < 1e-6) | (s > 10) | (s < 1e-6) | (s > 10)
        s = np.clip(s, 1e-6, 10)        
        s = np.log10(s)
        fix_pixels(s, bad_data)

        if mode == 'zone':
            savg_zone += s
            bd_zone |= bad_data
        elif mode == 'full':
            savg_full += s
            bd_full |= bad_data
        elif mode == 'both':
            savg_full += s
            savg_zone += s[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
            bd_full |= bad_data
            bd_zone |= bad_data[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]

    if mode in ('zone', 'both'):
        tnsr_zone[..., product_index] = anisotropic_diffusion(savg_zone / len(sigmaAvgNames), params[0], params[1], 0.2, option=1)
    if mode in ('full', 'both'):
        tnsr_full[..., product_index] = anisotropic_diffusion(savg_full / len(sigmaAvgNames), params[0], params[1], 0.2, option=1)
    product_index += 1


if ('sigma_hypot' in products) or ('sigma_pol' in products):
    if 'sigma_hypot' in products:
        params = products['sigma_hypot']
    else:
        params = products['sigma_pol']
        
    for svvn, svhn in zip(sigmaVVNames, sigmaVHNames):
        print(svvn, svhn)
        svv = envi.load(svvn)[0]
        svh = envi.load(svhn)[0]
        
        if mode == 'zone':
            svv = svv[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
            svh = svh[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
        
        bad_data = (svv < 1e-6) | (svv > 10) | (svh < 1e-6) | (svh > 10)
        svh = np.clip(svh, 1e-6, 10)
        sv = np.clip(np.hypot(svv, svh), 1e-6, 10)
        
        svpol = None
        if 'sigma_pol' in products:
            svpol = np.arcsin(svh / sv)
            fix_pixels(svpol, bad_data)
            svpol = gaussian_filter(svpol, params[2])
            svpol = anisotropic_diffusion(svpol, params[3], params[4], 0.2, option=1)
        svv = None
        svh = None
        
        sv = np.log10(sv)
        fix_pixels(sv, bad_data)
        sv = anisotropic_diffusion(sv, params[0], params[1], 0.2, option=1)
        
        if mode == 'zone':
            if 'sigma_hypot' in products:
                tnsr_zone[..., product_index] = sv
                product_index += 1
            if 'sigma_pol' in products:
                tnsr_zone[..., product_index] = svpol
                product_index += 1
            bd_zone |= bad_data
        elif mode == 'full':
            if 'sigma_hypot' in products:
                tnsr_full[..., product_index] = sv
                product_index += 1
            if 'sigma_pol' in products:
                tnsr_full[..., product_index] = svpol
                product_index += 1
            bd_full |= bad_data
        elif mode == 'both':
            if 'sigma_hypot' in products:
                tnsr_full[..., product_index] = sv
                tnsr_zone[..., product_index] = sv[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
                product_index += 1
            if 'sigma_pol' in products:
                tnsr_full[..., product_index] = svpol
                tnsr_zone[..., product_index] = svpol[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
                product_index += 1
            bd_full |= bad_data
            bd_zone |= bad_data[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]


if ('coh' in products):
    params = products['coh']
    for cn in cohNames:
        print(cn)
        c = envi.load(cn)[0]

        if mode == 'zone':
            c = c[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
        
        bad_data = (c < 0) | (c > 1) | (c < 0) | (c > 1)
        c = np.clip(c, 0, 1)
        
        fix_pixels(c, bad_data)
        c = anisotropic_diffusion(c, params[0], params[1], 0.2, option=1)
        
        if mode == 'zone':
            tnsr_zone[..., product_index] = c
            product_index += 1
            bd_zone |= bad_data
        elif mode == 'full':
            tnsr_full[..., product_index] = c
            product_index += 1
            bd_full |= bad_data
        elif mode == 'both':
            tnsr_full[..., product_index] = c
            tnsr_zone[..., product_index] = c[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
            product_index += 1
            bd_full |= bad_data
            bd_zone |= bad_data[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]

if ('coh_avg' in products):
    if mode in ('zone', 'both'):
        cavg_zone = np.zeros(zone_shape, dtype=np.float32)
    if mode in ('full', 'both'):
        cavg_full = np.zeros(full_shape, dtype=np.float32)
    params = products['coh_avg']

    for cn in cohAvgNames:
        print(cn)
        c = envi.load(cn)[0]

        if mode == 'zone':
            c = c[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
        
        bad_data = (c < 0) | (c > 1) | (c < 0) | (c > 1)
        c = np.clip(c, 0, 1)
        
        fix_pixels(c, bad_data)

        if mode == 'zone':
            cavg_zone += c
            bd_zone |= bad_data
        elif mode == 'full':
            cavg_full += c
            bd_full |= bad_data
        elif mode == 'both':
            cavg_full += c
            cavg_zone += c[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
            bd_full |= bad_data
            bd_zone |= bad_data[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]

    if mode in ('zone', 'both'):
        tnsr_zone[..., product_index] = anisotropic_diffusion(cavg_zone / len(cohAvgNames), params[0], params[1], 0.2, option=1)
    if mode in ('full', 'both'):
        tnsr_full[..., product_index] = anisotropic_diffusion(cavg_full / len(cohAvgNames), params[0], params[1], 0.2, option=1)
    product_index += 1


if ('coh_hypot' in products) or ('coh_pol' in products):
    if 'coh_hypot' in products:
        params = products['coh_hypot']
    else:
        params = products['coh_pol']

    for cvvn, cvhn in zip(cohVVNames, cohVHNames):
        print(cvvn, cvhn)
        cvv = envi.load(cvvn)[0]
        cvh = envi.load(cvhn)[0]
        
        if mode == 'zone':
            cvv = cvv[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
            cvh = cvh[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
        
        bad_data = (cvv < 0) | (cvv > 1) | (cvh < 0) | (cvh > 1)
        cvh = np.clip(cvh, 0, 1)
        cv = np.clip(np.hypot(cvv, cvh), 0, 2)
        
        cvpol = None
        if 'coh_pol' in products:
            cvpol = np.arcsin(cvh / cv)
            fix_pixels(cvpol, bad_data)
            cvpol = gaussian_filter(cvpol, params[2])
            cvpol = anisotropic_diffusion(cvpol, params[3], params[4], 0.2, option=1)
        cvv = None
        cvh = None
        
        fix_pixels(cv, bad_data)
        cv = anisotropic_diffusion(cv, params[0], params[1], 0.2, option=1)
        
        if mode == 'zone':
            if 'coh_hypot' in products:
                tnsr_zone[..., product_index] = cv
                product_index += 1
            if 'coh_pol' in products:
                tnsr_zone[..., product_index] = cvpol
                product_index += 1
            bd_zone |= bad_data
        elif mode == 'full':
            if 'coh_hypot' in products:
                tnsr_full[..., product_index] = cv
                product_index += 1
            if 'coh_pol' in products:
                tnsr_full[..., product_index] = cvpol
                product_index += 1
            bd_full |= bad_data
        elif mode == 'both':
            if 'coh_hypot' in products:
                tnsr_full[..., product_index] = cv
                tnsr_zone[..., product_index] = cv[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
                product_index += 1
            if 'coh_pol' in products:
                tnsr_full[..., product_index] = cvpol
                tnsr_zone[..., product_index] = cvpol[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]
                product_index += 1
            bd_full |= bad_data
            bd_zone |= bad_data[zone[0][0]:zone[1][0], zone[0][1]:zone[1][1]]

                
if mode in ('zone', 'both'):
    np.save('tnsr_zone.npy', tnsr_zone)
    np.save('bd_zone.npy', bd_zone)
if mode in ('full', 'both'):
    np.save('tnsr_full.npy', tnsr_full)
    np.save('bd_full.npy', bd_full)
    
system("say 'assembling complete'")
