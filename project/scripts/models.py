import torch.nn as nn
import torch.nn.functional as F
from utility import logMsg


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG13_m': [2, 2, 'M', 8, 8, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M']
}


class CellVGG(nn.Module):
    def __init__(self, vgg_name):
        super(CellVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 4
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=(1,3,3), padding=(0,1,1)),
                           nn.ReLU(inplace=True)]
                in_channels = x
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
