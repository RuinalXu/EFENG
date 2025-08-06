import torch
import torch.nn as nn
from transformers import BertModel
from layers import MLP, MaskAttention, CoAttention, SelfAttention
from torch.nn import Sequential, Linear, ReLU
from configurator import config


class EFENGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_bert = BertModel.from_pretrained(config.bert_path).requires_grad_(False)
        for name, parameter in self.content_bert.named_parameters():
            if name.startswith('encoder.layer.11'):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        self.rationale_bert = BertModel.from_pretrained(config.bert_path).requires_grad_(False)
        for name, parameter in self.rationale_bert.named_parameters():
            if name.startswith('encoder.layer.11'):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        self.co_attention = CoAttention(hidde_size=config.model.mlp.dims[-1], attention_size=64)
        self.self_attention = SelfAttention(multi_head_num=1, input_size=config.emb_dim)
        self.mask_attention = MaskAttention(input_dim=config.emb_dim)
        self.aggregator = MaskAttention(input_dim=config.aggregator_dim)
        self.mlp = MLP(input_dim=config.aggregator_dim, hidden_dim=config.model.mlp.dims, dropout=config.model.mlp.dropout)
        self.rationale_attention = MaskAttention(input_dim=config.emb_dim)
        self.rationale_mlp = Sequential(Linear(in_features=config.emb_dim, out_features=config.model.mlp.dims[-1]),
                                        ReLU(),
                                        Linear(in_features=config.model.mlp.dims[-1], out_features=3))
        self.selector = NewsEmbeddingSelector(embed_dim=config.emb_dim)

    def forward(self, kwargs):
        # input
        content, content_mask = kwargs['content_token_ids'], kwargs['content_mask']
        rationale_, rationale_mask = kwargs['rationale_token_ids'], kwargs['rationale_mask']
        rationale = self.selector(rationale_)
        content_feature = self.content_bert(input_ids=content, attention_mask=content_mask)[0]
        rationale_feature = self.rationale_bert(input_ids=rationale, attention_mask=rationale_mask)[0]
        content_mask_att, _ = self.mask_attention(inputs=content_feature, mask=content_mask)
        rationale_mask_att, _ = self.mask_attention(inputs=rationale_feature, mask=rationale_mask)
        content_att, rationale_att, content_att_weight, rationale_att_weight = self.co_attention(new_batch=content_feature, entity_desc_batch=rationale_feature)
        content_rationale = torch.cat((content_att, rationale_att), dim=2)
        rationale_pred = self.rationale_mlp(self.rationale_attention(rationale_feature)[0]).squeeze(1)
        all_feature = torch.cat(
            tensors=(content_mask_att.unsqueeze(1), rationale_mask_att.unsqueeze(1), content_rationale),
            dim=2)
        final_feature, _ = self.aggregator(all_feature)
        label_pred = self.mlp(final_feature)
        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'final_feature': final_feature,
            'content_feature': content_mask_att,
            'rationale_pred': rationale_pred
        }
        return res


class NewsEmbeddingSelector(nn.Module):
    def __init__(self, embed_dim):
        super(NewsEmbeddingSelector, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.query = nn.Parameter(torch.randn(embed_dim))  # learnable query vector

    def forward(self, embeddings):
        scores = torch.matmul(embeddings, self.query)
        weights = self.softmax(scores)
        # Weighted sum
        weighted_embedding = torch.sum(weights.unsqueeze(-1) * embeddings, dim=0)
        return weighted_embedding, weights