import torch
import torch.nn as nn
import torch.nn.functional as F #interpolate
import json
import math

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

class SABlock(nn.Module):
    layer_idx = 0
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, bias=False, downsample=False, structure=[],layername=None):
        super(SABlock, self).__init__()

        channels = structure[SABlock.layer_idx][:-1]
        side = structure[SABlock.layer_idx][-1]
        SABlock.layer_idx += 1
        self.scales = [None, 2, 2, 2] #[None, 2, 2, 2]
        self.stride = stride

        self.downsample = None if downsample == False else \
                          nn.Sequential(nn.Conv1d(inplanes, planes * SABlock.expansion, kernel_size=1, stride=1, bias=bias),
                                        nn.BatchNorm1d(planes * SABlock.expansion))

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(planes)

        # kernel size == 1 if featuremap size == 1
        self.conv2 = nn.ModuleList([nn.Conv1d(planes, channels[i], kernel_size=3 if side / 2**i > 1 else 1, stride=1, padding=1 if side / 2**i > 1 else 0, bias=bias) if channels[i] > 0 else \
                                    None for i in range(len(self.scales))])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(channels[i]) if channels[i] > 0 else \
                                  None for i in range(len(self.scales))])

        self.conv3 = nn.Conv1d(sum(channels), planes * SABlock.expansion, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm1d(planes * SABlock.expansion)
        
        self.layername=layername
        if self.layername=='layer2' or self.layername=='layer3': 
            print('f'*20)
            self.attennet=eca_layer(planes * SABlock.expansion)
        else:
            print('g'*30)


    def forward(self, x):
        x = F.max_pool1d(x, self.stride, self.stride) if self.stride > 1 else x

        residual = self.downsample(x) if self.downsample != None else x

        out1 = self.conv1(x)
        out1 = F.relu(self.bn1(out1))

        out2_list = []
        # print(out1.size())
        size = [1,out1.size(2)]
        for i in range(len(self.scales)):
            out2_i = out1 # copy
            if self.scales[i] != None:
                out2_i = F.max_pool1d(out2_i, self.scales[i], self.scales[i])
            if self.conv2[i] != None:
                out2_i = self.conv2[i](out2_i)
            if self.scales[i] != None:
                # print(out2_i.shape)
                # nearest mode is not suitable for upsampling on non-integer multiples 
                # mode = 'nearest' if size[0] % out2_i.shape[1] == 0 and size[1] % out2_i.shape[2] == 0 else 'bilinear'
                mode = 'nearest' if size[1] % out2_i.shape[2] == 0  else 'bilinear'
                # print("ff1")
                out2_i=torch.unsqueeze(out2_i, 2)
                # print(out2_i.shape)
                # out2_i = F.upsample(out2_i, size=size, mode=mode)
                out2_i = F.interpolate(out2_i, size=size, mode=mode)
                # print("ff2")
                # print(out2_i.shape)
                out2_i=torch.squeeze(out2_i, 2)
                # print(out2_i.shape)
            if self.bn2[i] != None:
                out2_i = self.bn2[i](out2_i)
                # print(out2_i.shape)
                out2_list.append(out2_i)
                # print("ff3")
        out2 = torch.cat(out2_list, 1)
        out2 = F.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)

        if self.layername=='layer2' or self.layername=='layer3':
            # print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffddddddddddddd')
            out3 = self.attennet(out3)

        out3 += residual
        out3 = F.relu(out3)

        return out3


class ScaleNet(nn.Module):

    def __init__(self, block, layers, structure, num_classes=212):
        super(ScaleNet, self).__init__()

        self.inplanes = 64
        self.structure = structure

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=1,layername='layer0') #64
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,layername='layer1') #128
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,layername='layer2') #256
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,layername='layer3')
        # self.fc = nn.Linear(256 * block.expansion, num_classes)
        # self.fc = nn.Linear(32768, num_classes) #FFF
        # self.loss = nn.CrossEntropyLoss() #FFF

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1,layername=None):
        downsample = True if stride != 1 or self.inplanes != planes * block.expansion else False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, structure=self.structure,layername=layername))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, downsample=False, structure=self.structure,layername=layername))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.max_pool1d(x, 3, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())
        # x = F.adaptive_avg_pool1d(x, 64)
        x = x.view(x.size(0), -1)
        # print(x.size())
        # x = self.fc(x)

        return x


def scalenet50(structure_path, ckpt=None, **kwargs):
    layer = [1,1,1,1]  #[3, 4, 6, 3]
    structure = json.loads(open(structure_path).read())
    model = ScaleNet(SABlock, layer, structure, **kwargs)
    return model

