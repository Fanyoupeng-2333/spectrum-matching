import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


class eca_layer2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5,    padding=(k_size - 1) // 2, bias=False) 
        # self.conv3 = nn.Conv1d(1, 1, kernel_size=7,    padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        # print('x',x.size())
        y = self.avg_pool(x)
        # print(y.size())
        # Two different branches of ECA module
        y=y.transpose(-1, -2)
        # print(y.size())
        x1=y[:,0,:].unsqueeze(1)
        x2=y[:,1,:].unsqueeze(1)
        x3=y[:,2,:].unsqueeze(1)
        # print(x1.size())
        y1 = self.conv(x1).transpose(-1, -2)
        y2 = self.conv(x2).transpose(-1, -2)
        # y3 = self.conv(x3).transpose(-1, -2)
        y=y1+y2
        # print(y.size())
        # y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        tt=x * y.expand_as(x)
        # print(y.size(),tt.size(),'tt.size()')

        return tt#x * y.expand_as(x)

class eca_layer3(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5,    padding=(k_size - 1) // 2, bias=False) 
        self.conv3 = nn.Conv1d(1, 1, kernel_size=7,    padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        # print('x',x.size())
        y = self.avg_pool(x)
        # print(y.size())
        # Two different branches of ECA module
        y=y.transpose(-1, -2)
        # print(y.size())
        x1=y[:,0,:].unsqueeze(1)
        x2=y[:,1,:].unsqueeze(1)
        x3=y[:,2,:].unsqueeze(1)
        # print(x1.size())
        y1 = self.conv(x1).transpose(-1, -2)
        y2 = self.conv(x2).transpose(-1, -2)
        y3 = self.conv(x3).transpose(-1, -2)
        y=y1+y2+y3
        # print(y.size())
        # y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        tt=x * y.expand_as(x)
        # print(y.size(),tt.size(),'tt.size()')

        return tt#x * y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(2)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5,    padding=(k_size - 1) // 2, bias=False) 
        # self.conv3 = nn.Conv1d(1, 1, kernel_size=7,    padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        # print('x',x.size())
        y = self.avg_pool(x)
        # print(y.size())
        # Two different branches of ECA module
        y=y.transpose(-1, -2)
        # print(y.size())
        x1=y[:,0,:].unsqueeze(1)
        x2=y[:,1,:].unsqueeze(1)
        # x3=y[:,2,:].unsqueeze(1)
        # print(x1.size())
        y1 = self.conv(x1).transpose(-1, -2)
        y2 = self.conv(x2).transpose(-1, -2)
        # y3 = self.conv(x3).transpose(-1, -2)
        y=y1+y2
        # print(y.size())
        # y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        tt=x * y.expand_as(x)
        # print(y.size(),tt.size(),'tt.size()')

        return tt#x * y.expand_as(x)
    
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal',layername=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv1d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stype = stype
        self.scale = scale
        self.width  = width
        self.layername=layername

        print('ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
        if self.layername=='layers2': 
            print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
            self.attennet=eca_layer(planes * self.expansion)
        else:
            print('ggggggggggggggggggggggggggggggggggggggggg')
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.layername=='layers2':
            # print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffddddddddddddd')
            out = self.attennet(out)

        if self.downsample is not None:
            # print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffddddddddddddd')
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=212): #class cmp-512
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0],stride=1,layername='layers0')  #64
        self.layer2 = self._make_layer(block, 64 , layers[1], stride=2,layername='layers1') #128
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,layername='layers2') #256
        # self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc1 = nn.Linear(5000, num_classes)
        self.fc = nn.Linear(65536, num_classes) # 131072 65536 FFF
        # self.oout = nn.LogSoftmax(dim=1)#FFF

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,layername=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale,layername=layername))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale,layername=layername))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        
        # print(x.size())
        # x = self.avgpool(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        # x = self.fc(x)
        # x = self.oout(x)
        # print(x.size())

        return x

def res2netfan24(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    model = Res2Net(Bottle2neck, [1, 1, 2, 1], baseWidth = 26, scale = 4, **kwargs)
    
    return model


#https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes / 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes / 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, x,gamma=2,bias=1):
        super(eca_layer, self).__init__()
        # x: input features with shape [b, c, h, w]
        self.x=x
        self.gamma=gamma
        self.bias=bias
        b, c, h, w = x.size()
        t=int(abs((math.log(c,2)+self.bias)/self.gamma))
        k_size= t if t%2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        # feature descriptor on the global spatial information
        y = self.avg_pool(self.x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return self.x * y.expand_as(self.x)
