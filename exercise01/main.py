from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as skfilters

from sklearn.ensemble import RandomForestClassifier

def question02():
    
    #indices of used images
    imgNum = 5;
    ind = range(1, 1+imgNum)
    print(ind)
    
    #load images, note that the images may of various widths/heights
    train_imgs = []
    test_imgs = []
    
    train_gt = []
    test_gt = []
    
    cnt = 0
    
    for i in ind:
        fname = 'rgb/horse'+ str(i).zfill(3)+'.jpg'
        train_imgs.append(misc.imread(fname))
        
        fname = 'figure_ground/horse'+ str(i).zfill(3)+'.jpg'
        train_gt.append(misc.imresize(misc.imread(fname), size=(train_imgs[cnt].shape[0], train_imgs[cnt].shape[1])))
        train_gt[cnt] = np.clip(train_gt[cnt], a_min=0, a_max=1)
        
        fname = 'rgb/horse'+ str(i+30).zfill(3)+'.jpg'
        test_imgs.append(misc.imread(fname))
        
        fname = 'figure_ground/horse'+ str(i+30).zfill(3)+'.jpg'
        test_gt.append(misc.imresize(misc.imread(fname), size=(test_imgs[cnt].shape[0], test_imgs[cnt].shape[1])))
        test_gt[cnt] = np.clip(test_gt[cnt], a_min=0, a_max=1)
        
        #print(train_imgs[cnt].shape)
        #print(train_gt[cnt].shape)
        
        '''
        for m in range(test_gt[cnt].shape[0]):
            for n in range(test_gt[cnt].shape[1]):
                if test_gt[cnt][m,n] == 0:
                    test_imgs[cnt][m,n,:] = 0

        plt.figure()
        plt.imshow(test_imgs[cnt])
        plt.show()
        
        '''
        
        cnt+=1
    
    
    # initialize training samples and targets
    nsamples = 0
    nfeatures = 9 #gaussian, laplacian, median
    
    for i in range(0, len(train_imgs)):
        dim = train_imgs[i].shape;
        nsamples += dim[0] * dim[1]
        
        
    X = np.zeros(shape=(nsamples, nfeatures), dtype='float64')
    Y = np.zeros(shape=nsamples, dtype='int64')
    
    print(X.shape)
    print(Y.shape)
    
    #compute features
    cnt = 0
    for i in range(0,len(train_imgs)):
        
        img_r = train_imgs[i][:,:,0]
        img_g = train_imgs[i][:,:,1]
        img_b = train_imgs[i][:,:,2]
        
        size = train_imgs[i].shape[0] * train_imgs[i].shape[1];
        
        ft_r = skfilters.gaussian(img_r, sigma = 3).flatten()
        ft_g = skfilters.gaussian(img_g, sigma = 3).flatten()
        ft_b = skfilters.gaussian(img_b, sigma = 3).flatten()
        
        X[cnt:cnt+size, 0] = ft_r[:]
        X[cnt:cnt+size, 1] = ft_g[:]
        X[cnt:cnt+size, 2] = ft_b[:]
        Y[cnt:cnt+size] = train_gt[i].flatten()     
        
        ft_r = skfilters.laplace(img_r, ksize=3).flatten()
        ft_g = skfilters.laplace(img_g, ksize=3).flatten()
        ft_b = skfilters.laplace(img_b, ksize=3).flatten()
        
        X[cnt:cnt+size, 3] = ft_r[:]
        X[cnt:cnt+size, 4] = ft_g[:]
        X[cnt:cnt+size, 5] = ft_b[:]
        Y[cnt:cnt+size] = train_gt[i].flatten()
        
        ft_r = skfilters.median(img_r, np.ones((3,3))).flatten()
        ft_g = skfilters.median(img_g, np.ones((3,3))).flatten()
        ft_b = skfilters.median(img_b, np.ones((3,3))).flatten()
        
        X[cnt:cnt+size, 6] = ft_r[:]
        X[cnt:cnt+size, 7] = ft_g[:]
        X[cnt:cnt+size, 8] = ft_b[:]
        Y[cnt:cnt+size] = train_gt[i].flatten()
        
        cnt += size
        
        '''
        print(train_ft[i].dtype)
        print(train_ft[i].shape)
        plt.figure()
        plt.imshow(train_ft[i][:,:,0], cmap='gray')
        plt.show()
        '''
    
    rfc = RandomForestClassifier(n_estimators=5)
    rfc.fit(X, Y)
    scores = rfc.score(X, Y)
    
    print(scores)
    
    img = test_imgs[1]
    img_size = img.shape[0]*img.shape[1]
    
    data = np.zeros(shape=(img_size, nfeatures), dtype='float64')
    
    cnt = 0
    data[cnt:cnt+img_size, 0] = skfilters.gaussian(img[:,:,0], sigma=3).flatten()
    data[cnt:cnt+img_size, 1] = skfilters.gaussian(img[:,:,1], sigma=3).flatten()
    data[cnt:cnt+img_size, 2] = skfilters.gaussian(img[:,:,2], sigma=3).flatten()
    
    data[cnt:cnt+img_size, 3] = skfilters.laplace(img[:,:,0], ksize=3).flatten()
    data[cnt:cnt+img_size, 4] = skfilters.laplace(img[:,:,1], ksize=3).flatten()
    data[cnt:cnt+img_size, 5] = skfilters.laplace(img[:,:,2], ksize=3).flatten()
    
    data[cnt:cnt+img_size, 6] = skfilters.median(img[:,:,0], np.ones((3,3))).flatten()
    data[cnt:cnt+img_size, 7] = skfilters.median(img[:,:,1], np.ones((3,3))).flatten()
    data[cnt:cnt+img_size, 8] = skfilters.median(img[:,:,2], np.ones((3,3))).flatten()
    
    predicted = rfc.predict(data)
    print(img.shape)
    print(predicted.dtype)
    print(predicted.shape)
    print(np.max(predicted))
    
    predicted = predicted.reshape(img.shape[0], img.shape[1])
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(test_gt[1], cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(predicted, cmap='gray')
    plt.show()
    

def main():
    question02();

if __name__ == '__main__':
    main()
