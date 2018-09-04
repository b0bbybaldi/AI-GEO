#!/usr/bin/env python3
import numpy as np
from glob import glob
import os

datatypeDict = {1:np.uint8, 2:np.int16, 3:np.int32, 4:np.float32, 5:np.float64, 6:np.complex64, 9:np.complex128, 12:np.uint16, 13:np.uint32, 14:np.int64, 15:np.uint64}
datatypeDictInv = {np.dtype(v).name : k for k, v in datatypeDict.items()}

def load(path):
    header = open(path + '.hdr','r')
    hdict = dict([tuple(val[:-1].split(' = ')) for val in header.readlines()[1:] if len(val)>1])

    if int(hdict['bands']) == 1:
        imshape = (int(hdict['lines']), int(hdict['samples']))
    else:
        print('Multibands not implemented')
        exit(-1)

    datatype = datatypeDict[int(hdict['data type'])]
    load_datatype = datatype
    if int(hdict['byte order']):
        load_datatype = np.dtype(datatype).newbyteorder('>')
        
    arr = np.fromfile(path + '.img', load_datatype).astype(datatype)
    arr = arr.reshape(imshape)
    np.nan_to_num(arr, copy=False)
    return arr, hdict

def read_header(path):
    header = open(path + '.hdr','r')
    hdict = dict([tuple(val[:-1].split(' = ')) for val in header.readlines()[1:] if len(val)>1])

    if int(hdict['bands']) == 1:
        imshape = (int(hdict['lines']), int(hdict['samples']))
    else:
        print('Multibands not implemented')
        exit(-1)

    return imshape, hdict

def save(path, arr, map_info, coord_string, chnames = 'my_ch_name', desc='my description'):
    nbands = 0
    if len(arr.shape) == 2:
        nbands = 1
    elif len(arr.shape) == 3:
        nbands = arr.shape[2]
    else:
        print('Wrong arr shape')
        exit(-1)
        
    hdr = open(path + '.hdr', 'w')
    hdr.write('ENVI\n')
    hdr.write('description = ' + desc + '\n')
    hdr.write('samples = ' + str(arr.shape[1]) + '\n')
    hdr.write('lines = ' + str(arr.shape[0]) + '\n')
    hdr.write('bands = ' + str(nbands) + '\n')
    hdr.write('header offset = 0\n')
    hdr.write('file type = ENVI Standard\n')
    hdr.write('data type = ' + str(datatypeDictInv[arr.dtype.name]) + '\n')
    hdr.write('interleave = bip\n')
    hdr.write('byte order = 0\n')

    if isinstance(chnames, str):
        hdr.write('band names = { ' + chnames + ' }\n')
    elif isinstance(chnames, list):
        hdr.write('band names = { ' + ', '.join(chnames) + ' }\n')
    else:
        print('Wrong chnames format')
        exit(-1)
        
    hdr.write('map info = ' + map_info + '\n')
    hdr.write('coordinate system string = ' + coord_string + '\n')
    hdr.close()
    
    arr.tofile(path + '.img')
    
def locate(path='./'):
    return [os.path.split(os.path.splitext(el)[0])[1] for el in glob(path + '*.hdr')]
