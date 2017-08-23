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

#import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys
from datetime import datetime

import os
import logging, argparse
import time

from utility import *
from models import *

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

class SubtractMean(object):
    
    def __call__(self, sample):
        
        images = sample['images'].astype(np.float32)
        labels = sample['labels']
        indices = sample['index']
        
        # only subtract mean in image data
        for c in [0, 1]:
            images[c] -=np.mean(images[c])
            
        return {'images': images, 'labels': labels, 'index':indices}

def showImages(sample, use_gui):
    
    images, labels, index = sample['images'], sample['labels'], sample['index']
    
    images = images.numpy()
    
    images = images.transpose((0, 2, 3, 1))
    
    print('image dimension: {0}, {1}'.format(images.shape[1], images.shape[2]) )
    print('mininum: ', np.min(images[0, :, :, 0]))
    print('maximum: ', np.max(images[0, :, :, 0]))
    
    
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
    if use_gui:
        plt.show()
    else:
        plt.savefig('sample_{}.png'.format(index))


def showBatchImages(sampleBatch, use_gui):
    
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
    if use_gui:
        plt.show()

''' show an item/sample (with specified index) from a dataset'''
''' assume data already converted to pytorch tensor '''
def displaySample(set, index=0, use_gui=False):
    sample = set.__getitem__(index)
    showImages(sample, use_gui)
    
    

''' show a single (with specified index) '''
''' or continuous batch(es) from a dataloader'''
''' assumed data already converted to tensor '''
def displayBatch(loader, index=0, single=True, use_gui=False):
    
    for i_batch, sample_batch in enumerate(loader):
        
        if single:
            if i_batch == index:
                print(i_batch, print(' '.join('index: %5s' % sample_batch['index'][j] for j in range(sample_batch['index'].size()[0]))))
                showBatchImages(sample_batch, use_gui)
                break
        else:
            if i_batch >= index:
                print(i_batch, print(' '.join('index: %5s' % sample_batch['index'][j] for j in range(sample_batch['index'].size()[0]))))
                showBatchImages(sample_batch, use_gui)



def train(epoch, net, optim, lossFcn, loader, history, use_cuda):
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(loader):
        inputs, labels = data['images'], data['labels']
         
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        inputs, labels = Variable(inputs), Variable(labels)
        
        optim.zero_grad()
        
        outputs = net(inputs)
        loss = lossFcn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0] #loss is a Variable
        
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()
    
    running_loss /= (i+1)
    running_acc = correct / total

    history['train_loss'].append(running_loss)
    history['train_acc'].append(running_acc)
    
    logMsg('\ttrn_loss: {0:.3f}, trn_acc: {1:.3f}'.format(running_loss, running_acc), args.log)


def validate(epoch, net, lossFcn, loader, history, use_cuda):
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(loader):
        inputs, labels = data['images'], data['labels']
        
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        inputs, labels = Variable(inputs), Variable(labels)
        
        outputs = net(inputs)
        
        loss = lossFcn(outputs, labels)
        
        running_loss += loss.data[0] #loss is a Variable
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()

    running_loss /= (i+1)
    running_acc = correct / total
    
    history['val_loss'].append(running_loss)
    history['val_acc'].append(running_acc)
        
    logMsg('\tval_loss: {0:.3f}, val_acc: {1:.3f}'.format(running_loss, running_acc), args.log)


if __name__ == '__main__':
    
    ''' command line inputs '''
    parser = argparse.ArgumentParser(description='Pytorch Transition Classifier Training')
    
    parser.add_argument('--learning-rate', '-lr', default=0.001, type=float, dest='lr', help='learning rate')
    parser.add_argument('--checkpoint', '-cp', default='', type=str, dest='checkpoint', help='resume from given checkpoint')
    parser.add_argument('--no-cuda', default=True, action='store_false', dest='cuda', help='do not use cuda')
    parser.add_argument('--train-batch-size', '-tb', default=4, type=int, dest='train_batch_size', help='training batch size')
    parser.add_argument('--validate-batch-size', '-vb', default=4, type=int, dest='val_batch_size', help='testing batch size')
    parser.add_argument('--epochs', default=6, type=int, dest='epochs', help='number of training epochs')
    parser.add_argument('--optimizer', '-o', default='SGD', type=str, dest='optimizer', help='optimzer')
    parser.add_argument('--no-log', default=True, action='store_false', dest='log', help='no logging')
    parser.add_argument('--doNotSaveModel', default=True, action='store_false', dest='saveModel', help='do not save model')
    parser.add_argument('--save-interval', '-si', default=2, type=int, dest='save_interval', help='save the model at some epoch interval')
    parser.add_argument('--model-name', '-m', default=datetime.now().strftime('%Y_%m_%d_%H_%M'), type=str, dest='modelName', help='name of model to save')
    parser.add_argument('--gui', default=False, action='store_true', dest='gui', help='use gui to display graphs')
    parser.add_argument('--resume', '-r', default=False, action='store_true', dest='resume', help='resume training')
    parser.add_argument('--data', '-d', default='data/Fluo-N2DH-SIM-01-samples-2017-08-04-shuffled-2c.h5', type=str, dest='dataFile', help='name of data file')
    parser.add_argument('--weight-decay', '-wd', default=0, type=float, dest='wd', help='weight decay')
    parser.add_argument('--arch', '-a', default='VGG13_m', type=str, dest='arch', help='network architecture')
    
    ''' initialize variables '''
    args = parser.parse_args()
    epochs = args.epochs
    
    start_epoch = 0
    
    if args.log:
        logging.basicConfig(filename=args.modelName+'.log',level=logging.DEBUG)
    
    logMsg('Arguments', args.log)
    for arg in vars(args):
        logMsg('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)), args.log)
    
    if args.cuda:
        if torch.cuda.is_available():
            use_cuda = True
        else:
            use_cuda = False
            logMsg('No cuda device is available. Use CPU.', args.log)
    else:
        use_cuda = False
    
    ''' prepare data '''
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 
    # load data and perform transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset = CellDataset(args.dataFile, '.', train=True, split = 0.75, transform=transforms.Compose([ToTensor()])) #
    
    trainloader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    
    testset = CellDataset(args.dataFile, '.', train=False, split = 0.75, transform=transforms.Compose([ToTensor()])) #

    testloader = DataLoader(testset, batch_size=args.val_batch_size, shuffle=False, num_workers=2)
    
    displaySample(set=trainset, index=0, use_gui=False)
    
    ''' load already trained model '''
    if args.checkpoint:
        
        #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
        chkpt = torch.load(args.checkpoint+'.md.tar')
        
        ''' do something about checkpoint...'''
        start_epoch = chkpt['epoch']
        net = chkpt['arch']
        net.load_state_dict(chkpt['state_dict'])
        if chkpt['arch_cuda']:
            net.cpu()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd) #define how to update gradient
        optimizer.load_state_dict(chkpt['optimizer'])
        netHist = chkpt['history']
        
        ''' post analysis '''
        visualizeWeights(net.features[0].weight.data, args.gui, args.checkpoint, 'checkpoint_conv1')
        #plotStatistics(netHist, args.gui, args.checkpoint, 'checkpoint')
        
        if not args.resume:
            sys.exit(0)
        
        logMsg('Resume from epoch={0}'.format(start_epoch), args.log)
      
    else:
        #net = Net()
        net = CellVGG(args.arch)
        net.apply(weights_init)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd) #define how to update gradient
        netHist = {'train_loss':list(), 'train_acc':list(), 'val_acc':list(), 'val_loss':list()}
    
    ''' network info '''
    net.printInfo(args.log)
    
    if use_cuda:
        net.cuda()

    lossFcn = nn.CrossEntropyLoss() #loss function
    
    logMsg('Start training', args.log)
    begintime = time.time()
    
    ''' train for number of epochs '''
    for epoch in range(start_epoch, start_epoch+epochs):
        
        epoch_begin = time.time()
        
        logMsg('Epoch {0}:'.format(epoch+1), args.log)
        
        train(epoch, net, optimizer, lossFcn, trainloader, netHist, use_cuda)
        
        validate(epoch, net, lossFcn, testloader, netHist, use_cuda)
        
        logMsg('used {0:.3f} sec.'.format(time.time()-epoch_begin), args.log)
        
        # save model at some interval
        if (epoch+1) % args.save_interval == 0 and args.saveModel:
            saveCheckpoint(args.modelName, epoch+1, net, optimizer, netHist, args.cuda)
            logMsg('Checkpoint saved. Subtotal time used: {0:.3f} min'.format((time.time()-begintime)/60), args.log)
            # output plot to check 
            plotStatistics(netHist, False, args.modelName, 'epoch{0}'.format(epoch+1))
        
        
    logMsg('Finished training. Total time used: {0:.3f} min'.format((time.time()-begintime)/60), args.log)
    
    ''' always save the model again at the end '''
    if args.saveModel:
        saveCheckpoint(args.modelName, epoch+1, net, optimizer, netHist, args.cuda)
        logMsg('Checkpoint saved.', args.log)
    
    #convert back to cpu 
    if use_cuda:
        net.cpu()
        
    #visualizeWeights(net.conv1.weight.data, args.gui, args.modelName, 'conv1_weight')
    plotStatistics(netHist, args.gui, args.modelName, 'epoch{0}'.format(epoch+1))
    
    #displayBatch(testloader, single=False)
    
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

   
