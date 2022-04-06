import pickle
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from DisGANN import  DAGNN
from DisGANN_share import  DAGNN_share
from tqdm import tqdm

class Cor_loss(Module):
    def __init__(self,cor_weight,channels,hidden_size,channel_size):
        super(Cor_loss, self).__init__()
        self.channel_num = channels
        self.cor_weight =cor_weight
        self.hidden_size = hidden_size
        self.channel_size = channel_size

    def forward(self,embedding):#解耦任意factor对
        if self.cor_weight==0:
            return 0
        else:
            embedding = embedding.view(-1, self.hidden_size)
            embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
            cor_loss = torch.tensor(0,dtype = torch.float)
            for i in range(self.channel_num):
                for j in range(i+1,self.channel_num):
                    x=embedding_weight[i]
                    y=embedding_weight[j]
                    cor_loss = cor_loss+self._create_distance_correlation(x, y)
            b=  (self.channel_num+1.0)* self.channel_num/2
            cor_loss = self.cor_weight * torch.div(cor_loss,b)
        return cor_loss
    def forward_(self,embedding):#解耦相邻factor对（内存不足情况的次优解）
        if self.cor_weight==0:
            return 0
        else:
            embedding = embedding.view(-1,self.hidden_size)
            embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
            cor_loss = torch.tensor(0,dtype = torch.float)
            for i in range(self.channel_num-1):
                x=embedding_weight[i]
                y=embedding_weight[i+1]
                cor_loss = cor_loss+self._create_distance_correlation(x, y)
            b=  (self.channel_num+1.0)* self.channel_num/2
            cor_loss = self.cor_weight * torch.div(cor_loss,b)
        return cor_loss
    def _create_distance_correlation(self,x,y):
        zero = trans_to_cuda(torch.tensor(0,dtype=float))
        def _create_centered_distance(X,zero):
            r = torch.sum(torch.square(X),1,keepdim=True)
            X_t = torch.transpose(X,1,0)
            r_t = torch.transpose(r,1,0)
            D = torch.sqrt(torch.maximum(r-2*torch.matmul(X,X_t)+r_t,zero)+1e-8)
            D = D - torch.mean(D,dim=0,keepdim=True)-torch.mean(D,dim=1,keepdim=True)+torch.mean(D)
            return D

        def _create_distance_covariance(D1,D2,zero):
                n_samples = D1.shape[0]
                n_samples = torch.tensor(n_samples,dtype=torch.float)
                sum = torch.sum(D1*D2)
                sum = torch.div(sum,n_samples*n_samples)
                dcov=torch.sqrt(torch.maximum(sum,zero)+1e-8)
                return dcov

        D1 = _create_centered_distance(x,zero)
        D2 = _create_centered_distance(y,zero)

        dcov_12 = _create_distance_covariance(D1, D2,zero)
        dcov_11 = _create_distance_covariance(D1, D1,zero)
        dcov_22 = _create_distance_covariance(D2, D2,zero)

        dcor = torch.sqrt(torch.maximum(dcov_11 * dcov_22, zero))+1e-10
        dcor = torch.div(dcov_12,dcor)

        return dcor

class SessionGraph(Module):
    def __init__(self, opt, n_node): 
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize  
        self.n_node = n_node
        self.batch_size = opt.batchSize   
        self.nonhybrid = opt.nonhybrid   
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        if opt.weightshare == False:
            self.DAGNN = DAGNN(opt.in_dims, opt.channels, opt.c_dims, opt.iterations, opt.beta, opt.layer_num,opt.dropout)
        else:
            self.DAGNN = DAGNN_share(opt.in_dims, opt.channels, opt.c_dims, opt.iterations, opt.beta, opt.layer_num,
                               opt.dropout)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


        self.final_channels = opt.channels[-1]
        self.score_channel = opt.score_channel
        self.c_dims_hidden =opt.c_dims[-1]
        self.linear_o = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=True)
        self.linear_tw = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=True)
        self.linear_th = nn.Linear(self.c_dims_hidden, 1, bias=False)
        self.linear_tran = nn.Linear(self.c_dims_hidden * 2, self.c_dims_hidden, bias=True)

        
        self.in_dims_t = opt.in_dims[-1]
        self.channels_t=opt.channels[-1]
        self.linear_11 = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=True)
        self.linear_22 = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=True)
        self.linear_33 = nn.Linear(self.c_dims_hidden, 1, bias=False)
        self.linear_t_tran = nn.Linear(self.in_dims_t * 3, self.in_dims_t, bias=True)
        self.linear_t_tran_ = nn.Linear(self.in_dims_t, self.in_dims_t, bias=True)

        self.pos_embedding1 = nn.Embedding(200, self.c_dims_hidden)
        self.w_11 = nn.Parameter(torch.Tensor(2 * self.c_dims_hidden, self.c_dims_hidden))
        self.w_22 = nn.Parameter(torch.Tensor(self.c_dims_hidden, 1))
        self.glu11 = nn.Linear(self.c_dims_hidden, self.c_dims_hidden)
        self.glu22 = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=False)



        self.loss_function = nn.CrossEntropyLoss()  #交叉熵损失
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2) #Adam优化算法
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()   #初始化权重参数
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def compute_scores_1(self, hidden_, mask):

        hidden_channel = torch.chunk(hidden_,self.final_channels,dim=2)

        b_channel = self.DAGNN.rout_emb(self.embedding.weight[1:])
        scores=[]
        for hidden,b in zip(hidden_channel,b_channel):
            ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
            q1 = self.linear_o(ht).view(ht.shape[0], 1, ht.shape[1])
            q2 = self.linear_tw(hidden)
            alpha = self.linear_th(torch.sigmoid(q1 + q2))
            a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
            if not self.nonhybrid:
                a = self.linear_tran(torch.cat([a, ht], 1))

            score = torch.matmul(a, b.transpose(1, 0))
            score = score.view(score.shape[0],-1,1)
            scores.append(score)
        scores=torch.cat([scores[i] for i in range(len(scores))], dim=2)
        scores = torch.sort(scores,dim=2,descending=True)[0]
        #scores = scores[:,:,:self.score_channel]
        scores = torch.sum(scores,dim=2)
        return scores  #(100,309)
    def forward(self, inputs, A):

        hidden = self.embedding(inputs)
        cor_hidden = self.DAGNN.rout_emb_cor(hidden)

        hidden =self.DAGNN(A,hidden)

        return hidden,cor_hidden
    
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable



def forward_cor(model, i, data):  
    alias_inputs, A, items, mask, targets = data.get_slice(i)  
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())  
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden,cor_hidden= model(items, A)  
    get = lambda i: hidden[i][alias_inputs[i]]   
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  
    return targets, model.compute_scores_1(seq_hidden, mask),cor_hidden


def train_test_cor(model, train_data, test_data,cor_model):
    print('start training: ', datetime.datetime.now())
    model.train()# 指定模型为训练模式，计算梯度
    cor_model.train()
    total_loss = 0.0
    total_loss_2 = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i in tqdm(slices):
        model.optimizer.zero_grad()
        targets, scores ,hideen = forward_cor(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        hideen = trans_to_cuda(hideen)
        loss_1 = model.loss_function(scores, targets - 1)
        loss_2 = cor_model(hideen)
        loss = loss_1 + loss_2
        loss.backward() # 反向传播
        model.optimizer.step()
        total_loss_2+=loss_2
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tcor_Loss:\t%.3f' % total_loss_2)
    model.scheduler.step()
    print('start predicting: ', datetime.datetime.now())
    cor_model.eval()
    model.eval()  # 指定模型为计算模式
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores,_ = forward_cor(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
