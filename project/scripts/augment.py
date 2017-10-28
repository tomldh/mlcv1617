'''
Created on 26 Oct 2017

@author: fighterlin
'''

import h5py
import random
import numpy
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import gaussian_filter
import math

from skimage.util import random_noise
from skimage import transform
from skimage.transform import AffineTransform
from skimage import transform as tf



def normalization(input, mask):

    #inputGauss = gaussian_filter(input, sigma=3)
    
    inputMasked_0 = numpy.multiply(input, mask)
    
    inputMasked = inputMasked_0[0:266, 0:266, :]
    
    inputNonZero = inputMasked[numpy.nonzero(inputMasked)]
    
    inputMean = numpy.mean(inputNonZero[:])
    
    inputMax = numpy.amax(inputNonZero[:])
    
    inputMin = numpy.amin(inputNonZero[:])
    
    inputStd = 1
    
    # for normalization
    if math.fabs(inputMax-inputMean) < math.fabs(inputMin-inputMean):
        inputStd = math.fabs(inputMin-inputMean)
    else:
        inputStd = math.fabs(inputMax-inputMean)
    
    for k in range(inputMasked.shape[0]):
        for j in range(inputMasked.shape[1]):
            if inputMasked[k,j,0] > 0:
                inputMasked[k,j,0] = (inputMasked[k,j,0]-inputMean) / (inputStd+1e-7)  
                inputMasked[k,j,0] = (inputMasked[k,j,0]+1)*127+1
                #inputMasked[k,j,0] /= 255
    
    checkmin = numpy.amin(inputMasked[:])
    checkmax = numpy.amax(inputMasked[:])
    
    print('[normalization] min: {0}, max: {1}'.format(checkmin, checkmax))
    
    return inputMasked
    

if __name__ == '__main__':
    
    showFlag = True
    ops = { 'origin' : 1,
            'fliph' : 0,
            'flipv' : 0,
            'gauss1' : 0,
            'gauss3' : 0,
            'noise' : 0,
            'rot+10' : 1,
            'tra+10' : 0,
            'rot-10' : 1,
            'tra-10' : 0 }
    
    random.seed(10)
    
    f = h5py.File('../data/Fluo-N2DH-SIM-01-samples-2017-08-04.h5', 'r')
    fop = h5py.File('../data/Fluo-N2DH-SIM-01-samples-2017-08-04-shuffled-2c-aug.h5', 'w')
    
    indices = list(range(f['images'].shape[0]))
    random.shuffle(indices)
    
    dimages = []
    dlabels = []
    cnt = 0
    total = 0
    print(f['labels'].shape)
    
    for i in indices:
        
        img1a = f['images'][i][0].astype(numpy.float64) #original
        img2a = f['images'][i][1].astype(numpy.float64) #original
        
        if showFlag:
            plt.figure()
            plt.imshow(img1a[:,:,0], cmap='gray')
            plt.figure()
            plt.imshow(img2a[:,:,0], cmap='gray')
        
        img1m = f['images'][i][2] #mask
        img2m = f['images'][i][3] #mask

        img1_255 = normalization(img1a, img1m)
        img2_255 = normalization(img2a, img2m)
        
        img1_norm = img1_255 / 255 #
        img2_norm = img2_255 / 255 #
        
        if showFlag:
            print('[Original] img1_shape: {0}, img2_shape: {1}'.format(img1_255.shape, img2_255.shape))
            plt.figure()
            plt.imshow(img1_255[:,:,0], cmap='gray')
            plt.figure()
            plt.imshow(img2_255[:,:,0], cmap='gray')
        
        if ops['origin']:
            imgs_orig = numpy.stack((img1_255, img2_255), axis=0)
        
            dimages.append(imgs_orig)
            dlabels.append(f['labels'][i])
        
            total += 1
        
        # fliph
        if ops['fliph']:
            img1_fliph = numpy.fliplr(img1_255)
            img2_fliph = numpy.fliplr(img2_255)
        
            if showFlag:
                print('[fliph] img1_shape: {0}, img2_shape: {1}'.format(img1_fliph.shape, img2_fliph.shape))
            
                plt.figure()
                plt.imshow(img1_fliph[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_fliph[:,:,0], cmap='gray')
            
            imgs_fliph = numpy.stack((img1_fliph, img2_fliph), axis=0)
            
            dimages.append(imgs_fliph)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        # flipv
        if ops['flipv']:
            img1_flipv = numpy.flipud(img1_255)
            img2_flipv = numpy.flipud(img2_255)
        
            if showFlag:
                print('[flipv] img1_shape: {0}, img2_shape: {1}'.format(img1_flipv.shape, img2_flipv.shape))
                plt.figure()
                plt.imshow(img1_flipv[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_flipv[:,:,0], cmap='gray')
        
            imgs_flipv = numpy.stack((img1_flipv, img2_flipv), axis=0)
        
            dimages.append(imgs_flipv)
            dlabels.append(f['labels'][i])
        
            total += 1
        
        #gaussian
        if ops['gauss1']:
            img1_gauss1 = gaussian_filter(img1_255, sigma=1)
            img2_gauss1 = gaussian_filter(img2_255, sigma=1)
            
            if showFlag:
                print('[gaussian] img1_shape: {0}, img2_shape: {1}'.format(img1_gauss1.shape, img2_gauss1.shape))
                plt.figure()
                plt.imshow(img1_gauss1[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_gauss1[:,:,0], cmap='gray')
            
            imgs_gauss1 = numpy.stack((img1_gauss1, img2_gauss1), axis=0)
            
            dimages.append(imgs_gauss1)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        #gaussian3
        if ops['gauss3']:
            img1_gauss3 = gaussian_filter(img1_255, sigma=3)
            img2_gauss3 = gaussian_filter(img2_255, sigma=3)
            
            if showFlag:
                print('[gaussian] img1_shape: {0}, img2_shape: {1}'.format(img1_gauss3.shape, img2_gauss3.shape))
                plt.figure()
                plt.imshow(img1_gauss3[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_gauss3[:,:,0], cmap='gray')
            
            imgs_gauss = numpy.stack((img1_gauss3, img2_gauss3), axis=0)
            
            dimages.append(imgs_gauss)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        # Noise
        if ops['noise']:
            img1_rn = normalization(random_noise(img1_norm, mode='gaussian', seed=0, var=0.01), img1m)
            
            img2_rn = normalization(random_noise(img2_norm, mode='gaussian', seed=0, var=0.01), img2m)
            
            if showFlag:
                print('[random noise] img1_shape: {0}, img2_shape: {1}'.format(img1_rn.shape, img2_rn.shape))
                plt.figure()
                plt.imshow(img1_rn[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_rn[:,:,0], cmap='gray')
            
            imgs_rn = numpy.stack((img1_rn, img2_rn), axis=0)
            
            dimages.append(imgs_rn)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        # rotate
        if ops['rot+10']:
            img1_rot5 = transform.rotate(img1_255, 10)
            img2_rot5 = transform.rotate(img2_255, 10)
            
            if showFlag:
                print('[rotation 5] img1_shape: {0}, img2_shape: {1}'.format(img1_rot5.shape, img2_rot5.shape))
                plt.figure()
                plt.imshow(img1_rot5[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_rot5[:,:,0], cmap='gray')
            
            imgs_rot5 = numpy.stack((img1_rot5, img2_rot5), axis=0)
            
            dimages.append(imgs_rot5)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        # translate
        if ops['tra+10']:
            img1_trans5 = tf.warp(img1_255, AffineTransform(translation=(10, 10)))
            img2_trans5 = tf.warp(img2_255, AffineTransform(translation=(10, 10)))
            
            if showFlag:
                print('[translation 5] img1_shape: {0}, img2_shape: {1}'.format(img1_trans5.shape, img2_trans5.shape))
                plt.figure()
                plt.imshow(img1_trans5[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_trans5[:,:,0], cmap='gray')
            
            imgs_trans5 = numpy.stack((img1_trans5, img2_trans5), axis=0)
            
            dimages.append(imgs_trans5)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        # rotate
        if ops['rot-10']:
            img1_rot10 = transform.rotate(img1_255, -10)
            img2_rot10 = transform.rotate(img2_255, -10)
            
            if showFlag:
                print('[rotation 5] img1_shape: {0}, img2_shape: {1}'.format(img1_rot10.shape, img2_rot10.shape))
                plt.figure()
                plt.imshow(img1_rot10[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_rot10[:,:,0], cmap='gray')
            
            imgs_rot10 = numpy.stack((img1_rot10, img2_rot10), axis=0)
            
            dimages.append(imgs_rot10)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        # translate
        if ops['tra-10']:
            img1_trans10 = tf.warp(img1_255, AffineTransform(translation=(-10, -10)))
            img2_trans10 = tf.warp(img2_255, AffineTransform(translation=(-10, -10)))
            
            if showFlag:
                print('[translation 10] img1_shape: {0}, img2_shape: {1}'.format(img1_trans10.shape, img2_trans10.shape))
            
                plt.figure()
                plt.imshow(img1_trans10[:,:,0], cmap='gray')
                plt.figure()
                plt.imshow(img2_trans10[:,:,0], cmap='gray')
            
            imgs_trans10 = numpy.stack((img1_trans10, img2_trans10), axis=0)
            
            dimages.append(imgs_trans10)
            dlabels.append(f['labels'][i])
            
            total += 1
        
        print('total: ', total, ', cnt: ', cnt, ', i: ', i, ', label: ', f['labels'][i])
        
        cnt += 1
        
        if showFlag:
            plt.show()
        
    
    print(cnt)
    print(total)
    
    ishape = (total, 2, 266, 266, 1)
    print('ishape: ', ishape)
    
    fop.create_dataset('images', ishape, numpy.float64, dimages, compression="gzip", compression_opts=9)
    fop.create_dataset('labels', (total,1), f['labels'][0].dtype, dlabels, compression="gzip", compression_opts=9)
    
    f.close()
    fop.close()
        

        
        