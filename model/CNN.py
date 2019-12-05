import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import re


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN,self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)

        self.embed.weight = nn.Parameter(args.word_embedding, requires_grad=False)#parameter可以理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)


    def forward(self, x):
        x = self.embed(x) # (N, W, D)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1) # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]# [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]# [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x) # (N, len(Ks)*Co)

        logit = self.fc1(x) # (N, C)

        return logit