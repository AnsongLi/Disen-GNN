import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DisConv(nn.Module):
    def __init__(self, in_dim, channels, C_dim, iterations, beta ,weight_list,bias_list): 
        super(DisConv, self).__init__()
        self.channels = channels
        self.in_dim = in_dim
        self.c_dim = C_dim
        self.iterations = iterations
        self.beta = beta
        self.weight_list = []
        self.bias_list = []
        self.relu = nn.ReLU()
        self.weight_list =weight_list
        self.bias_list = bias_list

        self.hidden_size = C_dim
        self.input_size = C_dim * 2
        self.gate_size = 3 * C_dim
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size),requires_grad=True)
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size),requires_grad=True)
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size),requires_grad=True)
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size),requires_grad=True)

        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size),requires_grad=True)
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size),requires_grad=True)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_1 = nn.Linear(self.in_dim, self.channels*self.c_dim, bias=True)
        self.linear_2 = nn.Linear(self.channels*self.c_dim, self.channels*self.c_dim, bias=True)
        self.linear_3 = nn.Linear(self.channels*self.c_dim, 1, bias=False)

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        c_features = []
        for i in range(self.channels):  
            z = torch.matmul(features, self.weight_list[i]) + self.bias_list[i]
            z = F.normalize(z, dim=2)
            c_features.append(z)
        out_features = c_features  
        for l in range(self.iterations):
            c_attentions_in = []
            c_attentions_out = []
            for i in range(self.channels):
                channel_f = out_features[i]
                attention_matrix_in, attention_matrix_out = self.parse_attention(adj, channel_f)
                c_attentions_in.append(attention_matrix_in)
                c_attentions_out.append(attention_matrix_out)
            all_attentions_in = torch.cat([c_attentions_in[i] for i in range(len(c_attentions_in))], dim=3)
            all_attentions_out = torch.cat([c_attentions_out[i] for i in range(len(c_attentions_out))], dim=3)
            all_attentions_in = F.normalize(all_attentions_in, dim=2,p=1)
            all_attentions_out = F.normalize(all_attentions_out, dim=2,p=1)
            for k in range(self.channels):
                atte_in = all_attentions_in[:, :, :, k].squeeze()
                atte_out = all_attentions_out[:, :, :, k].squeeze()
                hidden = out_features[k]  # 目标点 zuk，也可以表示目标点
                input_in = torch.matmul(atte_in,self.linear_edge_in(hidden)) + self.b_iah  
                input_out = torch.matmul(atte_out,self.linear_edge_out(hidden)) + self.b_oah  
                out_features_now = torch.cat([input_in, input_out], 2)
                gi = F.linear(out_features_now, self.w_ih, self.b_ih)
                gh = F.linear(hidden, self.w_hh, self.b_hh)
                i_r, i_i, i_n = gi.chunk(3, 2) 
                h_r, h_i, h_n = gh.chunk(3, 2)
                resetgate = torch.sigmoid(i_r + h_r)  
                inputgate = torch.sigmoid(i_i + h_i)  
                newgate = torch.tanh(i_n + resetgate * h_n)
                hy = newgate + inputgate * (out_features[k] - newgate)
                out_features[k]=hy
        output = torch.cat([out_features[i] for i in range(len(out_features))], dim=2)
        q1 =  self.linear_1(features)
        q2 =  self.linear_2(output)
        alpha = self.linear_3(torch.sigmoid(q1+q2))
        output = alpha*features + (1-alpha)*output
        return output
    
    def route_embedding(self,embedding):
        c_features = []
        for i in range(self.channels): 
            z = torch.matmul(embedding, self.weight_list[i]) + self.bias_list[i]
            z = F.normalize(z, dim=1)
            c_features.append(z)
        return c_features



    def parse_attention(self, adj, features):
        attention_matrix = torch.matmul(features, features.permute(0,2,1)) 
        neg_attention = torch.zeros_like(attention_matrix)
        adj_in = adj[:,:,:adj.shape[1]]
        adj_out = adj[:, :, adj.shape[1]:]
        attention_matrix_in = torch.where(adj_in > 0, attention_matrix, neg_attention)
        attention_matrix_out = torch.where(adj_out > 0, attention_matrix, neg_attention)
        attention_matrix_in = attention_matrix_in * adj_in
        attention_matrix_out = attention_matrix_out * adj_out
        attention_matrix_in = attention_matrix_in * 1.0 / (self.beta)
        attention_matrix_out = attention_matrix_out * 1.0 / (self.beta)
        attention_matrix_in = torch.unsqueeze(attention_matrix_in, dim=3)
        attention_matrix_out = torch.unsqueeze(attention_matrix_out, dim=3)
        return attention_matrix_in,attention_matrix_out

