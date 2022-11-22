import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """Siamese network with weighted L1 distance"""
    def __init__(self, embedding_network):
        super(SiameseNetwork, self).__init__()
        self.embedding_network = embedding_network
        self.embed_dim = 65536  #16512 # 8256 8192 65536 32768
        self.joint_layer = self._make_joint_layer(in_features=self.embed_dim, 
            out_features=1)        
        self.loss = nn.BCELoss() #nn.BCELoss() nn.CrossEntropyLoss()

    def embedding(self, x):
        x_t = self.embedding_network(x)
        # print(x_t.size())
        x_t = x_t.view(x_t.size()[0], -1)
        # print(x_t.size())
        return x_t

    def forward(self, input0, input1):
        x0_t = self.embedding(input0)
        x1_t = self.embedding(input1)
        diff = x0_t - x1_t
        # Weighted L1 loss
        diff = torch.abs(diff)
        output = self.joint_layer(diff)
        return output

    def _make_joint_layer(self, in_features, out_features):
        joint_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Sigmoid())
        return joint_layer

