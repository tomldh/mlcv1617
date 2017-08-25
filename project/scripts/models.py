import torch.nn as nn
import torch.nn.functional as F
from utility import logMsg


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG13_m': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'C1' : [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'C2' : [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']
}


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.random_()
        #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

class CellVGG(nn.Module):
    def __init__(self, vgg_name, ch):
        super(CellVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], ch)

        if vgg_name == 'VGG13_m':
            self.fc1 = nn.Linear(4096, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 2)
        
        elif vgg_name == 'VGG13':
            self.fc1 = nn.Linear(32768, 4096)
            self.bn1 = nn.BatchNorm1d(4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.fc3 = nn.Linear(1024, 2)
        
        elif vgg_name == 'C1':
            self.fc1 = nn.Linear(8192, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 2)
        
        elif vgg_name == 'C2':
            self.fc1 = nn.Linear(16384, 2048)
            self.bn1 = nn.BatchNorm1d(2048)
            self.fc2 = nn.Linear(2048, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 2)
        
        #self.dp = nn.Dropout(p=0.3)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        #out = self.dp(out)
        out = F.relu(self.bn2(self.fc2(out)))
        #out = self.dp(out)
        out = self.fc3(out)
        return out

    def _make_layers(self, cfg, ch):
        layers = []
        in_channels = ch
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=(1,3,3), padding=(0,1,1)),
                           nn.BatchNorm3d(x),
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
