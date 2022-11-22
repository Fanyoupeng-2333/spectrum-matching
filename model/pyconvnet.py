import torch
import torch.nn as nn
import os


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


class PyConv1d(nn.Module):
    """PyConv1d with padding (general case). Applies a 1d PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv1d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv1d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv1d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv1d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyConvBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1,layername=None):
        super(PyConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # self.layername=layername
        # if self.layername=='layer2' or self.layername=='layer3': 
        #     print('f'*20)
        #     self.attennet=eca_layer(planes * self.expansion)
        # else:
        #     print('g'*30)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # if self.layername=='layer2' or self.layername=='layer3':
        #     # print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffddddddddddddd')
        #     out = self.attennet(out)

        out += identity
        out = self.relu(out)

        return out


class PyConvBasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBasicBlock1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_pyconv(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=1,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvBasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = get_pyconv(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes * self.expansion)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PyConvResNet(nn.Module):

    def __init__(self, block, layers, num_classes=512, zero_init_residual=False, norm_layer=None, dropout_prob0=0.0):
        super(PyConvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2, norm_layer=norm_layer,    #64
                                       pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 2, 4, 8],layername='layer0') #[1, 4, 8, 16]
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, norm_layer=norm_layer,    #128
                                       pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 2, 4],layername='layer1')
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, norm_layer=norm_layer,   #256
                                       pyconv_kernels=[3, 5], pyconv_groups=[1, 2],layername='layer2')
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, norm_layer=norm_layer,   #512
                                       pyconv_kernels=[3], pyconv_groups=[1],layername='layer3')
        # self.avgpool = nn.AdaptiveAvgPool1d(16)

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(32768, num_classes)  #FFF

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PyConvBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, pyconv_kernels=[3], pyconv_groups=[1],layername=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer,
                            pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups,layername=layername))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups,layername=layername))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())

        if self.dp is not None:
            x = self.dp(x)

        # x = self.fc(x)

        return x


def pyconvresnet50(pretrained=False, **kwargs):
    model = PyConvResNet(PyConvBlock, [1, 1, 1, 1], **kwargs)
    return model


