import torch
import logging
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def logMsg(msg, use_log=False, printToConsole=True):
    
    if use_log:
        logging.debug(msg)
            
    if printToConsole:
        print(msg)
        
def plotStatistics(history, use_gui=False, fprefix='', fname=''):

    ''' plots '''
    plt.figure()
    plt.plot(range(len(history['train_loss'])), history['train_loss'], color='b', label='train_loss')
    plt.plot(range(len(history['val_loss'])), history['val_loss'], color='r', label='val_loss')
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim([0, 1])
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
    plt.ylim([0, 1])
    plt.legend(loc=4)
    
    if use_gui:
        plt.show()
    else:
        plt.savefig('{0}_{1}_train_val_acc.png'.format(fprefix, fname))

def visualizeWeights(weight, use_gui=False, fprefix='', fname=''):
    
    wnp = weight.numpy()
    vmin = np.amin(wnp)
    vmax = np.amax(wnp)
    
    
    #2d conv weight
    if wnp.ndim == 4:
        plt.figure(figsize=(wnp.shape[1],wnp.shape[0]))
        idx = 1
        for i in range(wnp.shape[0]):
            for j in range(wnp.shape[1]):
                ax = plt.subplot(wnp.shape[0], wnp.shape[1], idx)
                ax.axis('off')
                ax.imshow(wnp[i, j, :], cmap='gray', vmin=vmin, vmax=vmax, interpolation='bilinear')
                idx += 1
    
    elif wnp.ndim == 5:
        plt.figure(figsize=(wnp.shape[1]*wnp.shape[2],wnp.shape[0]))
        idx = 1
        for i in range(wnp.shape[0]):
            for j in range(wnp.shape[1]):
                for k in range(wnp.shape[2]):
                    ax = plt.subplot(wnp.shape[0]*wnp.shape[2], wnp.shape[1], idx)
                    ax.axis('off')
                    ax.imshow(wnp[i, j, k, :], cmap='gray', vmin=vmin, vmax=vmax, interpolation='bilinear')
                    idx += 1
          
        
    #plt.subplots_adjust(wspace=0, hspace=0)
    
    if use_gui:
        plt.show()
    else:
        plt.savefig('{0}_{1}.png'.format(fprefix, fname))
        
def saveCheckpoint(filename, epoch, net, optim, history, use_cuda):
    
    state = {'epoch': epoch,
             'arch': net,
             'arch_cuda': use_cuda,
             'state_dict': net.state_dict(), 
             'optimizer': optim.state_dict(),
             'history':history }
    
    torch.save(state, filename+'.md.tar')

