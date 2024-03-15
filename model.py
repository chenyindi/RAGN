import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_gcn import GAT, GCN, Rel_GAT
from model_ner import BERT_NER
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits, SelfAttention, PointwiseFeedForward, DynamicLSTM
from tree import *
from common.sublayer import MultiHeadedAttention
from pytorch_transformers.modeling_bert import BertPooler


class GLF(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(GLF, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.global_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout)
        self.local_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout)
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, hidden_size)
        # mask: (batch_size, seq_len)
        x = x.transpose(0, 1)
        global_out, _ = self.global_attn(x, x, x, key_padding_mask=mask)
        sub_len = self.num_heads
        sub_num = x.size(0) // sub_len
        x = x.reshape(sub_num, sub_len, -1, self.hidden_size)  # (sub_num, sub_len, batch_size, hidden_size)

        local_out = []
        for i in range(sub_num):
            sub_x = x[i]  # (sub_len, batch_size, hidden_size)
            sub_out, _ = self.local_attn(sub_x, sub_x, sub_x)  # (sub_len, batch_size, hidden_size)
            local_out.append(sub_out)

        local_out = torch.cat(local_out, dim=0)  # (seq_len, batch_size, hidden_size)

        fusion_out = torch.cat([global_out, local_out], dim=-1)  # (seq_len, batch_size, hidden_size * 2)
        fusion_out = self.fusion(fusion_out)  # (seq_len, batch_size, hidden_size)

        fusion_out = fusion_out.transpose(0, 1)

        return fusion_out


class GLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GLU, self).__init__()
        self.linear_v = nn.Linear(in_features, out_features)
        self.linear_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        v = self.linear_v(x)
        g = self.linear_g(x)
        out = F.glu(torch.cat([v, g], dim=-1), dim=-1)
        return out


class Aspect_Bert_GAT(nn.Module):
    '''
    R-GAT with bert
    '''

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Bert_GAT, self).__init__()
        self.args = args

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf=False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        self.mean_pooling_double = PointwiseFeedForward(args.embedding_dim * 2, args.embedding_dim, args.embedding_dim)
        self.bert_pooler = BertPooler(config)

        # bert4ner
        config_ner = BertConfig.from_pretrained(args.bert_model_dir)
        config_ner.num_labels = args.num_labels
        self.bert4ner = BERT_NER(BertModel.from_pretrained(
            args.bert_model_dir, config=config_ner, from_tf=False))

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]
        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        cat_num = sum([args.use_bert_global, args.use_gat_feature, args.use_cross_attn, args.use_ner_feature, args.use_cdw_feature])
        last_hidden_size = args.embedding_dim * cat_num
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        #
        self.cross_attn = MultiHeadedAttention(args.cross_attn_heads, 768)
        self.gtu = GLF(args.embedding_dim, args.cross_attn_heads)  # 3
        # self.glu = GLU(args.embedding_dim, args.embedding_dim)  # 2


    def feature_dynamic_weighted(self, text_local_indices, aspect_indices, distances_input=None):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.args.max_seq_len, self.args.bert_dim),
                                          dtype=np.float32)
        mask_len = self.args.SRD
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) - 2
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][2])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
                for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                    srd = abs(i - asp_avg_index) + asp_len / 2
                    if srd > self.args.SRD:
                        distances[i] = 1 - (srd - self.args.SRD) / np.count_nonzero(texts[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_input[text_i]
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(texts[text_i])
                    else:
                        distances_i[i] = 1
                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]

            masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
            return masked_text_raw_indices.to(self.args.device)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer, input_cat_ids, segment_ids, pos_class,
                dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs, ner_valid, ner_mask,
                ner_input_mask, ner_segment_ids, ncon_num_ids, adj, src_mask, context_attention_mask):
        # print('============////ncon_num_ids:', ncon_num_ids.size()) torch.Size([16, 128])
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:, 0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)
        bert_local_out = self.bert(input_ids, token_type_ids=segment_ids)[0]
        feature_output = outputs[0]  # (N, L, D)
        pool_out = outputs[1]  # (N, D)

        if self.args.local_context_focus == 'cdw':
            cdw_vec = self.feature_dynamic_weighted(input_ids, input_aspect_ids, ncon_num_ids)
            bert_local_output = torch.mul(bert_local_out, cdw_vec)
            # bert_local_output = self.conv1(bert_local_output)
            out_cat = torch.cat((bert_local_output, feature_output), dim=-1) # 残差归一化后86.43 不行
            mean_pool = self.mean_pooling_double(out_cat)
            pooled_out = self.bert_pooler(mean_pool)
        # pooled_out = pooled_out.unsqueeze(-1)
        # pool_out = pool_out.unsqueeze(-1)
        # pool = torch.cat((pool_out, pooled_out), dim=-1)
        # pool = pool.mean(dim=-1)

        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)

        ner_feature, ner_logit, ner_sequence = self.bert4ner(input_ids, attention_mask=ner_input_mask,
                                                             token_type_ids=ner_segment_ids,
                                                             valid_ids=ner_valid, label_mask=ner_mask)
        features = []
        if self.args.use_gat_feature:
            features.append(dep_out)
        if self.args.use_bert_global:
            features.append(pool_out)
        if self.args.use_ner_feature:
            features.append(ner_feature)
        if self.args.use_cross_attn:
            ner_sequence = self.gtu(ner_sequence)  # #######
            ner_sequence = ner_sequence.mean(dim=1)
            attn_feature = self.cross_attn(dep_out, ner_sequence, ner_sequence)[0]  # 进2出3
            attn_feature = attn_feature.mean(dim=1)
            features.append(attn_feature)
        if self.args.use_cdw_feature:
            features.append(pooled_out)

        feature_out = torch.cat(features, dim=1)  # (N, D')
        # feature_out = torch.cat([dep_out, ner_feature], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit, ner_logit


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
