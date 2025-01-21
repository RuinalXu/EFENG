import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, ModuleList
import math


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_layer=True):
        super().__init__()
        mlp_layers = list()
        dim_ = None
        for dim_ in hidden_dim:
            mlp_layers.append(Linear(in_features=input_dim, out_features=dim_))
            mlp_layers.append(ReLU())
            mlp_layers.append(Dropout(p=dropout))
            input_dim = dim_
        if output_layer:
            mlp_layers.append(Linear(in_features=dim_, out_features=1))
        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)


class MaskAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_layer = Linear(in_features=input_dim, out_features=1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class ScaledDotProductAttention(torch.nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, dim_model, dropout=0.1):
        super().__init__()
        assert dim_model % head_num == 0
        self.d_k = dim_model // head_num
        self.h = head_num
        self.hidden_layers = ModuleList([Linear(in_features=dim_model, out_features=dim_model) for _ in range(3)])
        self.output_layer = Linear(in_features=dim_model, out_features=dim_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.hidden_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_layer(x), attn


class SelfAttention(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(-1))
        feature, attn = self.attention(query=query, key=key, value=value, mask=mask)
        return feature, attn


class CoAttention(nn.Module):
    def __init__(self, hidde_size: int, attention_size: int):  # hidden_size:d, attention_size:k
        super(CoAttention, self).__init__()

        self.hidden_size = hidde_size
        self.Wl = nn.Parameter(torch.zeros(size=(hidde_size * 2, hidde_size * 2)), requires_grad=True)
        self.Ws = nn.Parameter(torch.zeros(size=(attention_size, hidde_size * 2)), requires_grad=True)
        self.Wc = nn.Parameter(torch.zeros(size=(attention_size, hidde_size * 2)), requires_grad=True)
        self.whs = nn.Parameter(torch.zeros(size=(1, attention_size)), requires_grad=True)
        self.whc = nn.Parameter(torch.zeros(size=(1, attention_size)), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.Wl.data.uniform_(-1.0, 1.0)
        self.Ws.data.uniform_(-1.0, 1.0)
        self.Wc.data.uniform_(-1.0, 1.0)
        self.whs.data.uniform_(-1.0, 1.0)
        self.whc.data.uniform_(-1.0, 1.0)

    def forward(self, new_batch, entity_desc_batch):
        # news_batch: [batch size, N, hidden size *2] hidden size:h
        S = torch.transpose(new_batch, 1, 2)
        # entity_desc_batch: [batch size, T, hidden size *2] T: entity description sentences for a news
        C = torch.transpose(entity_desc_batch, 1, 2)

        attF = torch.tanh(torch.bmm(torch.transpose(C, 1, 2), torch.matmul(self.Wl, S)))  # dim [batch_size,T,N]

        WsS = torch.matmul(self.Ws, S)  # dim[batch,a,N] a:attention size
        WsC = torch.matmul(self.Wc, C)  # dim[batch,a,T]

        Hs = torch.tanh(WsS + torch.bmm(WsC, attF))  # dim[batch,a,N]
        Hc = torch.tanh(WsC + torch.bmm(WsS, torch.transpose(attF, 1, 2)))  # dim[batch,a,T]

        a_s = F.softmax(torch.matmul(self.whs, Hs), dim=2)  # dim[batch,1,N]
        a_c = F.softmax(torch.matmul(self.whc, Hc), dim=2)  # dim[batch,1,T]

        s = torch.bmm(a_s, new_batch)  # dim[batch,1,2h]
        c = torch.bmm(a_c, entity_desc_batch)  # [batch,1,2h]
        return s, c, a_s, a_c
