'''

Machine Learning for Computer Vision 

Project: Transition Classifier by neural network

@author: Dehui Lin
'''

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import h5py

import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys

class CellDataset(Dataset):
    
    def __init__(self, hdf5File, root_dir, train=True, split=0.9, transform=None):
        self.hdf5File = hdf5File#h5py.File(hdf5File, 'r')
        self.root_dir = root_dir
        self.train = train
        self.split = split
        self.transform = transform
    
    def __len__(self):
        # need to open-close file object, otherwise error in h5py as follows
        # 'OSError: Can't read data (Inflate() failed)'
        f = h5py.File(self.hdf5File, 'r')
        nsamples = f['images'].shape[0]
        f.close()
        
        if self.train:
            return int(nsamples * self.split)
        else:
            return int(nsamples * (1-self.split))
    
    def __getitem__(self, index):
        # need to open-close file object, otherwise error in h5py as follows
        # 'OSError: Can't read data (Inflate() failed)'
        f = h5py.File(self.hdf5File, 'r')
        
        if self.train == False:
            nsamples = f['images'].shape[0]
            index += int(nsamples * self.split)
        
        images = f['images'][index]
        labels = f['labels'][index]
        f.close()
        
        sample = {'images':images, 'labels':labels, 'index':index}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    
    def __call__(self, sample):
        
        images = sample['images'].astype(np.float32)
        labels = sample['labels']
        indices = sample['index']
        
        images = images.transpose((0, 3, 1, 2))
        
        return {'images': torch.from_numpy(images), 'labels': int(labels), 'index':indices}
        
        

def showImages(images, labels, index):
    
    images = images.numpy()
    
    images = images.transpose((0, 2, 3, 1))
    
    print('mininum: ', np.max(images[0][:, :, 0]))
    
    assert images.ndim == 4
    
    channels = images.shape[0]
    x = images.shape[1]
    y = images.shape[2]
    z = images.shape[3] 
    
    nRows = z
    nCols = channels
    total = nRows * nCols
    
    pltIndex = 1
    
    plt.figure()
    
    for i in range(z):
            for j in range(channels):
                ax = plt.subplot(nRows, nCols, pltIndex)
                ax.axis('off')
                ax.set_title('Label {0}'.format(labels))
                ax.imshow(images[j][:, :, i], cmap='gray')
                
                pltIndex += 1


def showBatchImages(sampleBatch):
    
    images, labels = sampleBatch['images'], sampleBatch['labels']
    
    images = images.numpy()
    
    images = images.transpose((0, 1, 3, 4, 2))
    
    assert images.ndim == 5
    
    batch = images.shape[0]
    channels = images.shape[1]
    x = images.shape[2]
    y = images.shape[3]
    z = images.shape[4] 
    
    nRows = batch
    nCols = channels * z
    total = nRows * nCols
    
    pltIndex = 1
    
    plt.figure()
    
    for i in range(batch):
        for j in range(z):
            for k in range(channels):
                ax = plt.subplot(nRows, nCols, pltIndex)
                ax.axis('off')  
                ax.set_title('Label {0}'.format(labels[i]))
                ax.imshow(images[i, k, :, :, j], cmap='gray')
                pltIndex += 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(4, 6, (1,5,5)) # output channel "6" is the number of filters
        self.pool = nn.MaxPool3d((1,2,2), 2)
        self.conv2 = nn.Conv3d(6, 16, (1,5,5))
        self.fc1 = nn.Linear(16*63*63, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
    
    # for score calculation
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*63*63) #-1 is deduced from the other dimension(16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def totalSize(self, t):
        num = 1
        for s in t.size():
            num *= s
        return num
            
    
    def printNetInfo(self):

        print('==============================================')
        print('Layer info (Not architecture)')
        for name, m in self.named_children():
            print('{0: <20}: {1}'.format(name, m))
        #print('==============================================')
        print('\n')
        #print('==============================================')
        print('Parameter Summary:')
        sumParam = 0
        for name, param in self.named_parameters():
            sumParam += self.totalSize(param)
            print('{0: <20}: {1: <10} -> {2}'.format(name, self.totalSize(param), param.size()))
        
        print('Total parameters: {}'.format(sumParam))
        print('==============================================\n')
    

def displaySample(index=0):
    sample = trainset.__getitem__(index)
    showImages(**sample)
    plt.show()
    


def displayBatch(loader, index=0, single=True):
    
    for i_batch, sample_batch in enumerate(loader):
        
        if single:
            if i_batch == index:
                print(i_batch, print(' '.join('index: %5s' % sample_batch['index'][j] for j in range(sample_batch['index'].size()[0]))))
                showBatchImages(sample_batch)
                plt.show()
                break
        else:
            if i_batch >= index:
                print(i_batch, print(' '.join('index: %5s' % sample_batch['index'][j] for j in range(sample_batch['index'].size()[0]))))
                showBatchImages(sample_batch)
                plt.show()
            

if __name__ == '__main__':
    
    trainset = CellDataset('Fluo-N2DH-SIM-01-samples-2017-08-04.h5', '.', train=True, split = 0.75, transform=transforms.Compose([ToTensor()])) #
    
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    
    testset = CellDataset('Fluo-N2DH-SIM-01-samples-2017-08-04.h5', '.', train=False, split = 0.75, transform=transforms.Compose([ToTensor()])) #

    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    displayBatch(testloader, single=False)
    
    '''
    print(trainset.__len__())

    for i in range(len(trainset)):
        
        if i > 1200: # and i % 100 == 99:
            sample = trainset.__getitem__(i)
            print(type(sample))
            print(i, sample['images'].size())
            print(i, type(sample['images']), type(sample['labels']))
            print('min_img ', sample['images'][0].max())
            showImages(**sample) #for **kwargs
            plt.show()
           
    '''
    
    '''
    
    # define cnn
    net = Net()
    
    lossFcn = nn.CrossEntropyLoss() #loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #define how to update gradient
    
    # start training
    for epoch in range(1):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data['images'], data['labels']
            #print(inputs.size(), ' ', labels.size())
            
            inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = lossFcn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0] #loss is a Variable
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 20))
                running_loss = 0.0
    
    print('Finished training')
    
    # testing
    dataiter = iter(testloader)
    sample = dataiter.next()
    images, labels, indices = sample['images'], sample['labels'], sample['index']
    
    print(' '.join('index: %5s' % indices[j] for j in range(4)))
    print(' '.join('label: %5s' % labels[j] for j in range(4)))
    
    outputs = net(Variable(images)) # output is (nsamples index in minibatch, nclasses score)
    print(outputs)
    _, predicted_idx = torch.max(outputs.data, 1)
    
    print(predicted_idx.size())
    
    print('Predicted: ', ' '.join('%5s' % predicted_idx[j][0] for j in range(4)))
    
    showBatchImages(sample)
    plt.show()
    
    '''
   