import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from layer_share import *

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

class DAGNN_share(nn.Module):
    def __init__(self, in_dim, channels, C_dim, iterations, beta, layer_num, dropout
                 ): 
        super(DAGNN_share, self).__init__()
        self.dropout_rate = dropout
        self.channels = channels
        self.C_dim = C_dim
        self.iterations = iterations
        self.beta = beta
        self.layer_num = layer_num

        self.in_dim = in_dim
        self.c_dim = C_dim
        self.weight_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(self.in_dim[0], self.c_dim[0]), dtype=torch.float), requires_grad=True) for i in
            range(self.channels[0]))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.c_dim[0]), dtype=torch.float), requires_grad=True) for i in
            range(self.channels[0]))
        DisAGNN = []
        for x in range(len(channels)):
            disconv = DisConv(in_dim[x], channels[x], C_dim[x], iterations, beta,self.weight_list,self.bias_list)
            DisAGNN.append(disconv)
        self.DisAGNN = nn.Sequential(*DisAGNN)

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)


    def forward(self, adj, features):
        h = features
        for i in range(len(self.DisAGNN)-1):
            h = self.DisAGNN[i](adj, h)
            h = F.dropout(h, self.dropout_rate, training=self.training)
        h = self.DisAGNN[len(self.DisAGNN)-1](adj, h)
        return h

    def rout_emb(self,emb):
        return (self.DisAGNN[len(self.DisAGNN)-1].route_embedding(emb))

    def rout_emb_cor(self,emb):
        cor_hidden=trans_to_cuda(torch.tensor([]))
        for i in range(len(self.DisAGNN[0].route_embedding(emb))):
            cor_hidden=torch.cat((cor_hidden,self.DisAGNN[0].route_embedding(emb)[i]),2)

        return cor_hidden

