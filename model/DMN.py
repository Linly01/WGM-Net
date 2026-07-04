import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math


class DMN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(DMN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))

        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.fc1=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.time_dim))]))
        self.fc2=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(self.time_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.middle_dim, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.hyperGNN_dim, dim_in))]))
        # 定义了一个可学习的记忆矩阵 P
        self.node_embeddings = nn.Parameter(torch.randn(10, self.time_dim), requires_grad=True)
        self.dropout = nn.Dropout(p=0.1)

        # self.fc3= nn.Linear(dim_in, dim_out, bias=True)
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        filter1 = self.fc1(x)    #用全连接层 MLP(x) 提取动态信号
        nodevec1 = filter1       #nodevec1 = Fi = MLP(x)

        # self.node_embeddings = 记忆矩阵 P,这行代码是 : nodevec2 = Pt = P × Tt
        nodevec2 = torch.mul(node_embeddings[0].unsqueeze(-2),self.node_embeddings) #[B,N,dim_in]

        cata = self.fc2(nodevec2)  # 将 Pt 进行线性变换。cata = Pt

        # 计算 Fi 和 Pt 的相似度权重
        supports2 = torch.softmax(torch.matmul(nodevec1, nodevec2.transpose(-2, -1)), dim=-1)
        # 进行加权，将权重和模式 Pt加权，得到具有流量模式特征的交通模式矩阵
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, cata)
        # 利用残差连接将原始的交通特征和得到的记忆矩阵拼接
        x_g = torch.stack([x,x_g2],dim=1)

        # 将 node_embeddings[1]和 weights_pool相乘，得到权重
        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]


        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        # 通过爱因斯坦求和公式计算x_g和weights的乘积，加上bias，得到隐藏特征 Ht = x_gconv
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        # x_gconv =self.fc3(x)

        return x_gconv

    @staticmethod
    def get_laplacian(graph,normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L