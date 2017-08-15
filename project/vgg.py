'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from utility import logMsg


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def totalSize(self, t):
        num = 1
        for s in t.size():
            num *= s
        return num
    
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