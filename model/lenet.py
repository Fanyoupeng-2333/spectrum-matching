import torch.nn as nn
import math

default_settings = {
    'ConvLayers': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'Conv1x1Layers':[2048],
    'TensorChannels': 1,
}


def LeNet(settings=default_settings):
    """ Shortcut to PyramidLeNet
    """
    model = PyramidLeNet(settings=settings)
    return model


class PyramidLeNet(nn.Module):
    """Pyramid-shaped LeNets"""
    def __init__(self, settings):
        super(PyramidLeNet, self).__init__()
        self.settings = settings
        self.features = self._build_conv_layers()        

    def forward(self, x):
        out = self.features(x) 
        # print("hello")       
        out = out.view(out.size()[0], -1)
        return out

    def _build_conv_layers(self):
        conv_layers = []
        in_channels = 1
        self.kernel_size = 21
        for x in self.settings['ConvLayers']:
            if x == 'M':
                conv_layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.kernel_size = math.ceil(self.kernel_size*0.5)
                if self.kernel_size==6:
                    self.kernel_size =5
            else:
                conv_layers += [nn.Conv1d(in_channels, x, 
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=math.floor(self.kernel_size*0.5))]
                conv_layers += [nn.BatchNorm1d(x)]
                conv_layers += [nn.LeakyReLU(0.1,inplace=True)] #nn.ReLU(inplace=True)
                in_channels = x
        return nn.Sequential(*conv_layers)


class LeNetClassifier(nn.Module):
    def __init__(self, settings, num_classes):
        super(LeNetClassifier, self).__init__()
        self.settings = settings
        # self.num23 = 1
        self.features = PyramidLeNet(settings=settings)
        self.num_classes = num_classes
        self.embedding_dim = 16512
        self.head = self._make_head()
        self.loss = nn.CrossEntropyLoss()
        
        print("LeNetClassifier LeNetClassifier LeNetClassifier LeNetClassifier")

    def forward(self, x):
        out = self.features(x) 
        # print(out.size())       
        # print(self.num23, out.size())
        # self.num23 = self.num23 + 1   
        out = out.view(out.size()[0], -1)
        out = self.head(out)
        return out

    def _make_head(self):
        head = []
        head.append(nn.Linear(self.embedding_dim, self.num_classes))
        # head.append(nn.BatchNorm1d(self.num_classes))
        head.append(nn.LogSoftmax(dim=1))
        print("world")
        return nn.Sequential(*head)




