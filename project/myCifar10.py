'''
Created on 2 Aug 2017

@author: fighterlin
'''

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys
from _collections import OrderedDict

from datetime import datetime

import os
import logging, argparse
import time

from utility import logMsg
from vgg import *

# function to show an image
def imshow(img, use_gui=False):
    if use_gui:
        plt.figure()
        img = img / 2 + 0.5 # un-normalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0))) # previously tensor stores (channel, width, height)
        plt.show()


        
# inherit nn.module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # output channel "6" is the number of filters
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        
    # for score calculation
    def forward(self, x):
            
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16*5*5) #-1 is deduced from the other dimension(16*5*5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
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
    
    def architecture(self):
        
        return OrderedDict((name, m) for (name, m) in self.named_children())
                    
    
    def printInfo(self, use_log=False):

        logMsg('==============================================', use_log)
        logMsg('Module info (Not architecture)', use_log)
        for name, m in self.named_children():
            logMsg('{0: <20}: {1}'.format(name, m), use_log)
        #print('==============================================')
        logMsg('\n', use_log)
        #print('==============================================')
        logMsg('Parameter Summary:', use_log)
        sumParam = 0
        for name, param in self.named_parameters():
            sumParam += self.totalSize(param)
            logMsg('{0: <20}: {1: <10} -> {2}'.format(name, self.totalSize(param), param.size()), use_log)
        
        logMsg('Total parameters: {}'.format(sumParam), use_log)
        logMsg('==============================================\n', use_log)

def visualizeWeights(weight, use_gui=False, fprefix='', fname=''):
    
    wnp = weight.numpy()
    vmin = np.amin(wnp)
    vmax = np.amax(wnp)
    plt.figure(figsize=(wnp.shape[1],wnp.shape[0]))
    
    #2d conv weight
    if wnp.ndim == 4:
        idx = 1
        for i in range(wnp.shape[0]):
            for j in range(wnp.shape[1]):
                ax = plt.subplot(wnp.shape[0], wnp.shape[1], idx)
                ax.axis('off')
                ax.imshow(wnp[i, j, :], cmap='gray', vmin=vmin, vmax=vmax, interpolation=None)
                idx += 1
                
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if use_gui:
        plt.show()
    else:
        plt.savefig('{0}_{1}.png'.format(fprefix, fname))
    

def train(epoch, net, optim, loader, history, use_cuda):
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(loader):
        inputs, labels = data
         
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


def validate(epoch, net, loader, history, use_cuda):
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(loader):
        inputs, labels = data
        
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

def saveCheckpoint(filename, epoch, net, optim, history, use_cuda):
    
    state = {'epoch': epoch,
             'arch': net,
             'arch_cuda': use_cuda,
             'state_dict': net.state_dict(), 
             'optimizer': optim.state_dict(),
             'history':history }
    
    torch.save(state, filename+'.md.tar')
    
def plotStatistics(history, use_gui=False, fprefix='', fname=''):

    ''' plots '''
    plt.figure()
    plt.plot(range(len(history['train_loss'])), history['train_loss'], color='r', label='train_loss')
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc=1)
    
    if use_gui:
        plt.show()
    else:
        plt.savefig('{0}_{1}_train_loss.png'.format(fprefix, fname))
    
    plt.figure()
    plt.plot(history['train_acc'], color='b', label='train_acc')
    plt.plot(history['val_acc'], color='r', label='val_acc')
    plt.title('Training/Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc=4)
    
    if use_gui:
        plt.show()
    else:
        plt.savefig('{0}_{1}_train_val_acc.png'.format(fprefix, fname))
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch myCifar10 Training')
    
    parser.add_argument('--learning-rate', '-lr', default=0.001, type=float, dest='lr', help='learning rate')
    parser.add_argument('--checkpoint', '-cp', default='', type=str, dest='checkpoint', help='resume from given checkpoint')
    parser.add_argument('--no-cuda', default=True, action='store_false', dest='cuda', help='do not use cuda')
    parser.add_argument('--batch-size', '-b', default=4, type=int, dest='batch_size', help='batch size')
    parser.add_argument('--epochs', default=6, type=int, dest='epochs', help='number of training epochs')
    parser.add_argument('--optimizer', '-o', default='SGD', type=str, dest='optimizer', help='optimzer')
    parser.add_argument('--no-log', default=True, action='store_false', dest='log', help='no logging')
    parser.add_argument('--doNotSaveModel', default=True, action='store_false', dest='saveModel', help='do not save model')
    parser.add_argument('--save-interval', '-si', default=2, type=int, dest='save_interval', help='save the model at some epoch interval')
    parser.add_argument('--model-name', '-m', default=datetime.now().strftime('%Y_%m_%d_%H_%M'), type=str, dest='modelName', help='name of model to save')
    parser.add_argument('--gui', default=False, action='store_true', dest='gui', help='use gui to display graphs')
    parser.add_argument('--resume', '-r', default=False, action='store_true', dest='resume', help='resume training')
    
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
        
    # load data and perform transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', transform=transform);
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2) # each item contains a batch of 4 images and labels
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    logMsg('Training and testing data loaded successfully.', args.log)
    
    '''
    # show a sample
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    '''
    
    if args.checkpoint:
        
        #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
        chkpt = torch.load(args.checkpoint+'.md.tar')
        
        ''' do something about checkpoint...'''
        
        start_epoch = chkpt['epoch']
        logMsg('Resume from epoch={0}'.format(start_epoch), args.log)
        print('arch_cuda: ', chkpt['arch_cuda'])
        net = chkpt['arch']
        net.load_state_dict(chkpt['state_dict'])
        if chkpt['arch_cuda']:
            net.cpu()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9) #define how to update gradient
        optimizer.load_state_dict(chkpt['optimizer'])
        netHist = chkpt['history']
        
        ''' post analysis '''
        visualizeWeights(net.conv1.weight.data, args.gui, args.checkpoint, 'checkpoint_conv1')
        plotStatistics(netHist, args.gui, args.checkpoint, 'checkpoint')
        
        if not args.resume:
            sys.exit(0)
        
    else:
        #net = Net()
        net = VGG('VGG16')
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9) #define how to update gradient
        netHist = {'train_loss':list(), 'train_acc':list(), 'val_acc':list(), 'val_loss':list()}
    
    net.printInfo(args.log)
    
    if use_cuda:
        net.cuda()

    lossFcn = nn.CrossEntropyLoss() #loss function
    
    logMsg('Start training', args.log)
    begintime = time.time()
    
    for epoch in range(start_epoch, start_epoch+epochs):
        
        epoch_begin = time.time()
        
        logMsg('Epoch {0}:'.format(epoch+1), args.log)
        
        train(epoch, net, optimizer, trainloader, netHist, use_cuda)
        
        validate(epoch, net, testloader, netHist, use_cuda)
        
        logMsg('used {0:.3f} sec.'.format(time.time()-epoch_begin), args.log)
        
        if (epoch+1) % args.save_interval == 0 and args.saveModel:
            saveCheckpoint(args.modelName, epoch+1, net, optimizer, netHist, args.cuda)
            logMsg('Checkpoint saved. Subtotal time used: {0:.3f} min'.format((time.time()-begintime)/60), args.log)
            # output plot to check 
            plotStatistics(netHist, False, args.modelName, 'epoch{0}'.format(epoch+1))
        
        
    logMsg('Finished training. Total time used: {0:.3f} min'.format((time.time()-begintime)/60), args.log)
    
    if args.saveModel:
        saveCheckpoint(args.modelName, epoch+1, net, optimizer, netHist, args.cuda)
        logMsg('Checkpoint saved.', args.log)
    
    #convert back to cpu 
    if use_cuda:
        net.cpu()
        
    #visualizeWeights(net.conv1.weight.data, args.gui, args.modelName, 'conv1_weight')
    plotStatistics(netHist, args.gui, args.modelName, 'epoch{0}'.format(epoch+1))
    
    pass