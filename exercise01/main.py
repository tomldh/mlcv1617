'''

MLCV exercise 01
Authors: Cristian Alexa, Dehui Lin

'''

from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as skfilters

from sklearn.ensemble import RandomForestClassifier

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
        
        # map intensity from [0, 255] to [0, 1]
        test_gt[cnt] = np.clip(test_gt[cnt], a_min=0, a_max=1)
        
        cnt+=1
    
    
    nsamples = 0
    nfeatures = 18 #gaussian, laplacian, median
    
    # calculate the size of nsample array (i.e. total number of pixels of all training images)
    for i in range(0, len(train_imgs)):
        img_height = train_imgs[i].shape[0]
        img_width = train_imgs[i].shape[1]
        nsamples += img_width * img_height
        
    # init sample array [nsamples, nfeatures]
    X = np.zeros(shape=(nsamples, nfeatures), dtype='float64')
    # init target array [nsamples, category]
    Y = np.zeros(shape=nsamples, dtype='int64')
    
    
    start_idx = 0
    end_idx = 0
    
    # fill in sample arrray and target array
    for i in range(0,len(train_imgs)):
        
        img_r = train_imgs[i][:,:,0]
        img_g = train_imgs[i][:,:,1]
        img_b = train_imgs[i][:,:,2]
        
        numOfImgPix = train_imgs[i].shape[0] * train_imgs[i].shape[1]; # total number of pixels of current image
        
        end_idx = start_idx + numOfImgPix #ending index in sample array for current image
        
        # apply different filters to R, G, B channels separately and store the features to corresp. columns
        ft_rg = skfilters.gaussian(img_r, sigma = 3).flatten()
        X[start_idx:end_idx, 0] = ft_rg
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
        
        # fill in the corresp. target category for each pixel
        Y[start_idx:end_idx] = train_gt[i].flatten()
        
        '''
        #plot computed featue image
        
        if i==2:
            plt.figure()
            plt.subplot(1,4,1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('initial picture')
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
        start_idx += numOfImgPix #update the beginning index in sample arrayfor next image
    
    
    # create random forest model
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X, Y)
    scores = rfc.score(X, Y)
    print('score:', scores)
    
    
    print('Index\tHorseTotal\tHorsePixelRate\tBgdTotal\tBgdPixelRate\n')
    
    # prepare test data
    
    for i in range(len(test_imgs)):
        
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
        # plot predictions
        
        plt.figure()
        
        plt.subplot(1,4,1)
        plt.title('horse'+ str(test_idx[i]).zfill(3)+'.jpg')
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
    
    for m in range(len(truth)):
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
    

def main():
    question02();

if __name__ == '__main__':
    main()
