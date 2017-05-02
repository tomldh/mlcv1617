'''

MLCV exercise 01
Authors: Cristian Alexa, Dehui Lin

'''
'''
from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as skfilters

from sklearn.ensemble import RandomForestClassifier

'''

import numpy

def question02():
    
    #index of training and testing images
    train_idx = [1, 2, 5, 7, 9, 10, 13, 14, 15, 21]
    test_idx = [75, 79, 80, 86, 87, 88, 89, 96, 109, 110]
    
    #load images, note that the images may of various widths/heights
   
    # list of rgb images
    train_imgs = []
    test_imgs = []
    
    # list of ground truth images
    train_gt = []
    test_gt = []

    # list of predicted images
    list_pred = []
    
    cnt = 0 #for indexing in list
    
    for i in train_idx:
        
        fname = 'rgb/horse'+ str(i).zfill(3)+'.jpg'
        train_imgs.append(misc.imread(fname))
        
        fname = 'figure_ground/horse'+ str(i).zfill(3)+'.jpg'
       
        # resize the ground truth image to the same size as the rgb image
        train_gt.append(misc.imresize(misc.imread(fname), size=(train_imgs[cnt].shape[0], train_imgs[cnt].shape[1])))
        
        # map intensity from [0, 255] to [0, 1]
        train_gt[cnt] = np.clip(train_gt[cnt], a_min=0, a_max=1)
        
        cnt+=1
    
    cnt = 0
    
    for i in test_idx:
        
        fname = 'rgb/horse'+ str(i).zfill(3)+'.jpg'
        test_imgs.append(misc.imread(fname))
        
        fname = 'figure_ground/horse'+ str(i).zfill(3)+'.jpg'
        
        # resize the ground truth image to the same size as the rgb image
        test_gt.append(misc.imresize(misc.imread(fname), size=(test_imgs[cnt].shape[0], test_imgs[cnt].shape[1])))
        
       
        test_gt[cnt] = np.clip(test_gt[cnt], a_min=0, a_max=1)# map intensity from [0, 255] to [0, 1]
        
        cnt+=1
    
    
    nsamples = 0
    nfeatures = 18 #gaussian, laplacian, median
    
    
    for i in range(0, len(train_imgs)): # calculate the size of nsample array
        img_height = train_imgs[i].shape[0] # (i.e. total number of pixels of all training images)
        img_width = train_imgs[i].shape[1]
        nsamples += img_width * img_height
        
    X = np.zeros(shape=(nsamples, nfeatures), dtype='float64') # init sample array [nsamples, nfeatures]
    Y = np.zeros(shape=nsamples, dtype='int64') # init target array [nsamples, category]
    
    
    start_idx = 0
    end_idx = 0
    
    
    for i in range(0,len(train_imgs)): # fill in sample arrray and target array
        
        img_r = train_imgs[i][:,:,0]
        img_g = train_imgs[i][:,:,1]
        img_b = train_imgs[i][:,:,2]
        
        numOfImgPix = train_imgs[i].shape[0] * train_imgs[i].shape[1]; # total number of pixels of current image
        
        end_idx = start_idx + numOfImgPix #ending index in sample array for current image
        
        
        ft_rg = skfilters.gaussian(img_r, sigma = 3).flatten() # apply different filters to R, G, B channels separately and 
        X[start_idx:end_idx, 0] = ft_rg # store the features to corresp. columns
        ft_gg = skfilters.gaussian(img_g, sigma = 3).flatten()
        X[start_idx:end_idx, 1] = ft_gg
        ft_bg = skfilters.gaussian(img_b, sigma = 3).flatten()
        X[start_idx:end_idx, 2] = ft_bg  
        
        ft_rl = skfilters.laplace(img_r, ksize=3).flatten()
        X[start_idx:end_idx, 3] = ft_rl
        ft_gl = skfilters.laplace(img_g, ksize=3).flatten()
        X[start_idx:end_idx, 4] = ft_gl
        ft_bl = skfilters.laplace(img_b, ksize=3).flatten()
        X[start_idx:end_idx, 5] = ft_bl
        
        ft_rm = skfilters.median(img_r, np.ones((3,3))).flatten()
        X[start_idx:end_idx, 6] = ft_rm
        ft_bm = skfilters.median(img_g, np.ones((3,3))).flatten()
        X[start_idx:end_idx, 7] = ft_bm
        ft_gm = skfilters.median(img_b, np.ones((3,3))).flatten()
        X[start_idx:end_idx, 8] = ft_gm
        
        X[start_idx:end_idx, 9] = skfilters.gaussian(img_r, sigma = 5).flatten()
        X[start_idx:end_idx, 10] = skfilters.gaussian(img_g, sigma = 5).flatten()
        X[start_idx:end_idx, 11] = skfilters.gaussian(img_b, sigma = 5).flatten()    
        
        X[start_idx:end_idx, 12] = skfilters.laplace(img_r, ksize=5).flatten()
        X[start_idx:end_idx, 13] = skfilters.laplace(img_g, ksize=5).flatten()
        X[start_idx:end_idx, 14] = skfilters.laplace(img_b, ksize=5).flatten()
        
        X[start_idx:end_idx, 15] = skfilters.median(img_r, np.ones((5,5))).flatten()
        X[start_idx:end_idx, 16] = skfilters.median(img_g, np.ones((5,5))).flatten()
        X[start_idx:end_idx, 17] = skfilters.median(img_b, np.ones((5,5))).flatten()
       
        Y[start_idx:end_idx] = train_gt[i].flatten() # fill in the corresp. target category for each pixel
        
        '''  
        if i==2:
            plt.figure()
            plt.subplot(1,4,1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('initial picture') #plot computed feature image
            plt.imshow( train_imgs[i])
            plt.subplot(1,4,2)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel R') 
            plt.imshow(img_r,cmap= 'gray')
            plt.subplot(1,4,3)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel G')
            plt.imshow(img_g,cmap= 'gray')
            plt.subplot(1,4,4)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel B')
            plt.imshow(img_b,cmap= 'gray')
            plt.show()
            
            
            plt.figure()
            plt.subplot(1,3,1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel R')
            plt.imshow(np.reshape(ft_rg,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.subplot(1,3,2)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel G')
            plt.imshow(np.reshape(ft_gg,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.subplot(1,3,3)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel B')
            plt.imshow(np.reshape(ft_bg,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.show()
            
            
            plt.figure()
            plt.subplot(1,3,1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel R')
            plt.imshow(np.reshape(ft_rl,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.subplot(1,3,2)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel G')
            plt.imshow(np.reshape(ft_gl,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.subplot(1,3,3)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel B')
            plt.imshow(np.reshape(ft_bl,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.show()
            
            
            plt.figure()
            plt.subplot(1,3,1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel R')
            plt.imshow(np.reshape(ft_rm,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.subplot(1,3,2)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel G')
            plt.imshow(np.reshape(ft_gm,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.subplot(1,3,3)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('channel B')
            plt.imshow(np.reshape(ft_bm,(img_r.shape[0],img_r.shape[1] )),cmap='gray')
            plt.show()
        '''
        start_idx += numOfImgPix #update the beginning index in sample array for next image
    
    
    rfc = RandomForestClassifier(n_estimators=10) # create random forest model
    rfc.fit(X, Y)
    scores = rfc.score(X, Y)
    print('score:', scores)
    
    
    print('Index\tHorseTotal\tHorsePixelRate\tBgdTotal\tBgdPixelRate\n')
    
    
    
    for i in range(len(test_imgs)): # prepare test data
        
        img_r = test_imgs[i][:,:,0]
        img_g = test_imgs[i][:,:,1]
        img_b = test_imgs[i][:,:,2]
        
        numOfImgPix = test_imgs[i].shape[0]*test_imgs[i].shape[1]
        
        data = np.zeros(shape=(numOfImgPix, nfeatures), dtype='float64')
        
        data[:, 0] = skfilters.gaussian(img_r, sigma=3).flatten()
        data[:, 1] = skfilters.gaussian(img_g, sigma=3).flatten()
        data[:, 2] = skfilters.gaussian(img_b, sigma=3).flatten()
        
        data[:, 3] = skfilters.laplace(img_r, ksize=3).flatten()
        data[:, 4] = skfilters.laplace(img_g, ksize=3).flatten()
        data[:, 5] = skfilters.laplace(img_b, ksize=3).flatten()
        
        data[:, 6] = skfilters.median(img_r, np.ones((3,3))).flatten()          
        data[:, 7] = skfilters.median(img_g, np.ones((3,3))).flatten()
        data[:, 8] = skfilters.median(img_b, np.ones((3,3))).flatten()
        
        data[:, 9] = skfilters.gaussian(img_r, sigma=5).flatten()
        data[:, 10] = skfilters.gaussian(img_g, sigma=5).flatten()
        data[:, 11] = skfilters.gaussian(img_b, sigma=5).flatten()
        
        data[:, 12] = skfilters.laplace(img_r, ksize=5).flatten()
        data[:, 13] = skfilters.laplace(img_g, ksize=5).flatten()
        data[:, 14] = skfilters.laplace(img_b, ksize=5).flatten()
        
        data[:, 15] = skfilters.median(img_r, np.ones((5,5))).flatten()
        data[:, 16] = skfilters.median(img_g, np.ones((5,5))).flatten()
        data[:, 17] = skfilters.median(img_b, np.ones((5,5))).flatten()
        
        predicted = rfc.predict(data)
        
        predictedOnlyHorse = np.copy(predicted)
        
        det = compareResults(predictedOnlyHorse, test_gt[i].flatten())
        
        print(test_idx[i], '\t', det[0], '\t', det[1], '\t', det[2], '\t', det[3], '\n')
        
        
        '''
        plt.figure()
        
        plt.subplot(1,4,1)
        plt.title('horse'+ str(test_idx[i]).zfill(3)+'.jpg') # In this block we plot predictions   
        plt.axis('off')
        plt.imshow(test_imgs[i])
        
        plt.subplot(1,4,2)
        plt.title('Ground truth')
        plt.axis('off')
        plt.imshow(test_gt[i], cmap='gray')
        
        plt.subplot(1,4,3)
        plt.title('Prediction')
        plt.axis('off')
        predicted = predicted.reshape(test_imgs[i].shape[0], test_imgs[i].shape[1])
        plt.imshow(predicted, cmap='gray')
        
        plt.subplot(1,4,4)
        plt.title('Horse Only')
        plt.axis('off')
        predictedOnlyHorse = predictedOnlyHorse.reshape(test_imgs[i].shape[0], test_imgs[i].shape[1])
        plt.imshow(predictedOnlyHorse, cmap='gray')
        
        plt.show()
        
        '''

def compareResults(prediction, truth):
    
    if len(prediction) != len(truth):
        print('Error: comparing arrays with different sizes')                   
    
    totalHorsePix = 0
    totalBgdPix = 0
    
    rate = [0, 0, 0, 0]
    
    sumOfCorrectHorse = 0
    sumOfCorrectBgd = 0
    
    for m in range(len(truth)): #In this block are computed the accuracies of predicting 
        # pixel is horse 
        if truth[m] == 1:
            
            totalHorsePix += 1
            
            # when prediction is correct
            if prediction[m] == truth[m]:
                sumOfCorrectHorse += 1
         
        # pixel is background       
        elif truth[m] == 0:
            
            totalBgdPix += 1
            
            #when prediction is correct
            if prediction[m] == truth[m]:
                sumOfCorrectBgd += 1
            else:
                prediction[m] = 0
    
    rate[0] = totalHorsePix
    rate[1] = totalBgdPix
    rate[2] = 1.0 * sumOfCorrectHorse / totalHorsePix
    rate[3] = 1.0 * sumOfCorrectBgd / totalBgdPix
    
    if (totalHorsePix+totalBgdPix) != len(truth):
        print('Error: truth pixels value is unequal to 0 or 1')
    
    return rate

def potts(a, b):
    if a == b:
        return 0
    else:
        return 1

def iterated_conditional_modes(unaries, beta, labels = None):
    shape = unaries.shape[0:2]
    n_labels = unaries.shape[2]

    if labels is None:
        # return the index (0-bgd,1-horse)
        # of the best predicted label (higher probability)
        labels = numpy.argmin(unaries, axis=2)

    print(labels)
    
    continue_search = True
    while(continue_search):
        continue_search = False
        for x0 in range(1, shape[0]-1):
            for x1 in range (1, shape[1]-1):

                current_label = labels[x0, x1]
                min_energy = float('inf')
                best_label = None

                for l in range(n_labels):

                    # evaluate cost
                    energy = 0.0

                    # unary terms
                    # energy += TODO
                    energy += beta * unaries[x0, x1, l]

                    # pairwise terms
                    # energy += TODO
                    energy += (1-beta) * potts(labels[x0, x1], labels[x0-1, x1])
                    energy += (1-beta) * potts(labels[x0, x1], labels[x0+1, x1])
                    energy += (1-beta) * potts(labels[x0, x1], labels[x0, x1-1])
                    energy += (1-beta) * potts(labels[x0, x1], labels[x0, x1+1])

                    if energy < min_energy:
                        min_energy = energy
                        best_label = l

                if best_label != current_label:
                    labels[x0, x1] = best_label
                    continue_search = True
    return labels
    

def main():                                                   
    question02();

if __name__ == '__main__':

    shape = [3, 3]
    n_labels = 2

    # unaries
    unaries = numpy.random.rand(shape[0], shape[1], n_labels)

    #print(unaries)
    
    # convert the probability to energy values
    unaries = -numpy.log(unaries)

    #print(unaries)
    
    # regularizer strength
    beta = 0.01

    labels = iterated_conditional_modes(unaries, beta=beta)

    print('\n')
    print(labels)
    
    #main()
