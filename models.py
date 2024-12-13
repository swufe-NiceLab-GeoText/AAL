import json
import logging
import os
import joblib
import time
import numpy as np
from copy import deepcopy

import ot
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import List, Optional

from transformers import BertForTokenClassification, BertModel, BertTokenizer
from transformers import CONFIG_NAME, PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME
from transformers import AdamW as BertAdam
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities, performance_measure
from seqeval.scheme import IOB2, IOBES

from utils import load_file, filtered_tp_counts

logger = logging.getLogger(__file__)

class EntityTypes(object):

    def __init__(self, 
        types_path: str, 
        tagging_scheme: str, 
        num_centroids: int=15,
        entity_embedding_size: int=128, 
        min_num_supports: int=3, 
        support_sample_std: float=1.0, 
        uniform=True
    ):

        self.tags_list, self.num_tags = self.get_tag_list(tagging_scheme)
        self.tags_dict = {t:i for i, t in enumerate(self.tags_list)}
        self.id2tags = {i:t for i, t in enumerate(self.tags_list)}
        self.types, self.types_list = self.load_entity_types(types_path)
        self.types_dict = {t:i for i, t in enumerate(self.types_list)}

        self.labels_list = ['O'] # ["O", "B-x", "I-x", ...]
        self.label_map = {"O":0}   #{"B-x": id, ...}
        self.id2label = {0: "O"}
        self.label_tag_map = {0:0}
        self.label_type_map = {0:0}
        self.tagging_scheme = tagging_scheme

        idx = 1
        for tag in self.tags_list[1:]:
            for typ in self.types_list[1:]:
                label = tag + "-" + typ
                self.labels_list.append(label)
                self.label_map[label] = idx
                self.id2label[idx] = label
                self.label_tag_map[idx] = self.tags_dict[tag]
                self.label_type_map[idx] = self.types_dict[typ]
                idx += 1

        self.memory = {}
        self.memory_size = {}
        self.num_centroids = num_centroids
        self.min_centroids = 5
        self.min_num_supports = min_num_supports
        self.support_sample_std = support_sample_std
        self.embedding_size = entity_embedding_size
        self.uniform = uniform
        self.init_memory()

    def init_memory(self):
        for idx in range(len(self.types_list)):
            self.memory_size[idx] = 0
            self.memory[idx] = np.random.randn(self.num_centroids, self.embedding_size)

    def get_tag_list(self, tagging_scheme):

        if tagging_scheme == "BIOES":
            tags_list = ["O", "B", "I", "E", "S"]
        elif tagging_scheme == "BIO":
            tags_list = ["O", "B", "I"]
        else:
            tags_list = ["O", "I"]
        num_tags = len(tags_list)

        return tags_list, num_tags

    def load_entity_types(self, types_path: str):

        logger.info(types_path)
        types = load_file(types_path, "json")
        logger.info(types)
        types_list = [jj for ii in types.values() for jj in ii]
        if "O" in types_list:
            types_list.remove("O")
        types_list.insert(0, "O")
        logger.info("Load %d entity types from %s.", len(types_list), types_path)

        return types, types_list
    
    def get_task_labels(self, types):

        task_specific_labels = [0]
        for tag in self.tags_list[1:]:
            for typ in types:
                if typ == "O": continue
                task_specific_labels.append(self.label_map[tag + "-" + typ])

        return task_specific_labels

    def get_similar_memory(self, hiddens):

        device = hiddens.device
        D = hiddens.shape[-1]
        hiddens = hiddens.detach().cpu().numpy()
        num_hiddens = hiddens.shape[0]

        if num_hiddens < self.min_num_supports:
            aug_hiddens_list = [hiddens]
            for _ in range((self.min_num_supports - num_hiddens) // num_hiddens + 1):
                aug_hiddens_list.append(hiddens + \
                    np.random.randn(*hiddens.shape) * self.support_sample_std)
            aug_hiddens_embed = np.concatenate(aug_hiddens_list, axis=0)[:self.min_num_supports]
        else:
            aug_hiddens_embed = hiddens

        ot_list = []
        transp_hiddens_embed = []
        for typ in range(len(self.types_list)):
            if typ in self.memory and self.memory_size[typ] >= self.min_centroids:
                ot_dist, coupling = self.calculate_ot_matrix(self.memory[typ][:self.memory_size[typ]],
                                                             aug_hiddens_embed, uniform=self.uniform)
                # perform standard barycentric mapping
                transp = coupling / np.sum(coupling, axis=1)[:, None]
                # set nans to 0
                transp[~np.isfinite(transp)] = 0
                # compute transported samples
                transp_Xs = np.dot(transp, aug_hiddens_embed)

                ot_list.append(ot_dist)
                transp_hiddens_embed.append(transp_Xs)
            else:
                ot_list.append(1e5)
                transp_hiddens_embed.append(None)
        
        t = 1.0 / np.array(ot_list)
        t = torch.tensor(t).to(device).type(torch.float32)
        memory = [torch.tensor(m).to(device).type(torch.float32) if m is not None else None for m in transp_hiddens_embed]

        return t, memory
    
    def calculate_ot_matrix(self, X, Y, uniform=True):

        X_expand = np.expand_dims(X, axis=1)
        Y_expand = np.expand_dims(Y, axis=0)
        
        # # calculate cost and normalize it 
        C = np.sum((X_expand - Y_expand)**2, axis=-1)
        C /= float(np.max(C))

        if uniform:
            a = np.ones(X.shape[0]) / X.shape[0]
            b = np.ones(Y.shape[0]) / Y.shape[0]
        else:
            a_dist = np.sum((X - np.mean(X, axis=0, keepdims=True))**2, axis=-1)
            b_dist = np.sum((Y - np.mean(Y, axis=0, keepdims=True))**2, axis=-1)
            a = F.softmax(a_dist / float(np.max(a_dist)))
            b = F.softmax(b_dist / float(np.max(b_dist)))

        T_reg = ot.sinkhorn(a, b, C, 1e-1) # entropic regularized OT
        ot_dist = np.sum(C * T_reg)

        return ot_dist, T_reg
    
    def update_memory(self, entity_embed, entity_labels, types_list):
        
        label_list = [self.types_dict[t] for t in types_list]
        for i, label in zip(range(1, len(types_list)+1), label_list):
            embed = entity_embed[entity_labels == i].detach().cpu().numpy()
            ms = self.memory_size[label]
            if ms + embed.shape[0] <= self.num_centroids:
                self.memory[label][ms: ms + embed.shape[0]] = embed
                self.memory_size[label] = ms + embed.shape[0]
            else:
                if embed.shape[0] < self.num_centroids:
                    self.memory[label] = np.concatenate([self.memory[label][max(0, ms-(self.num_centroids-embed.shape[0])):ms], embed], axis=0)
                else:
                    self.memory[label] = embed[np.random.choice(np.arange(embed.shape[0]), self.num_centroids, replace=False)]
                self.memory_size[label] = self.num_centroids


class InferenceNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):

        super(InferenceNet, self).__init__()

        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim*2, hidden_dim)
        self.output_mu = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, input1, input2, gamma, device):

        input1_mean = torch.mean(input1, dim=0, keepdim=True) if input1 is not None else torch.zeros(1, self.input_dim).to(device)
        input2_mean = torch.mean(input2, dim=0, keepdim=True) if input2 is not None else torch.zeros(1, self.input_dim).to(device)
        inputs = torch.cat([input1_mean, input2_mean], dim=-1)
        hiddens = self.dropout(F.relu(self.layer1(inputs)))
        mu = gamma * self.output_mu(hiddens) + (1 - gamma) * input1_mean
        logstd = -10 * torch.ones_like(mu)

        return mu, logstd
    

class ProtoBertForTokenClassification(BertForTokenClassification):

    def __init__(self, *args, **kwargs):

        super(ProtoBertForTokenClassification, self).__init__(*args, **kwargs)

        self.input_size = self.config.hidden_size
        self.span_loss = nn.functional.cross_entropy
        self.type_loss = nn.functional.cross_entropy
        self.label_loss = nn.functional.cross_entropy
        self.dropout = nn.Dropout(p=0.1)
        self.log_softmax = nn.functional.log_softmax
        self.class_information = None
        self.dim = 32
        self.fu = nn.Linear(768, self.dim)
        self.fu.weight.data /= (self.dim / 32) ** 0.5
        self.fu.bias.data /= (self.dim / 32) ** 0.5
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.fs = nn.Linear(768, self.dim)
        self.fs.weight.data /= (self.dim / 32) ** 0.5
        self.fs.bias.data /= (self.dim / 32) ** 0.5
        self.line4 = nn.Linear(2, self.num_labels).to(0)
        self.line5 = nn.Linear(self.num_labels, self.num_labels).to(0)
        self.line6 = nn.Linear(self.num_labels, self.num_labels).to(0)
        self.ln3 = nn.LayerNorm(self.num_labels, 1e-5, True)
        self.activate = nn.Sigmoid()
        self.activate2 = nn.Sigmoid()
        self.entity_types_len = None
    def set_config(self, project_type_embedding: bool = True, type_embedding_size: int=128, sample_size: int=10, entity_types_len=None):
        
        self.project_type_embedding = project_type_embedding 
        self.sample_size = sample_size
        type_embedding_size = type_embedding_size if project_type_embedding else self.input_size

        if project_type_embedding:
            self.type_project = nn.Linear(self.input_size, type_embedding_size)

        self.inference_block = InferenceNet(input_dim=type_embedding_size, hidden_dim=type_embedding_size, \
            output_dim=type_embedding_size, dropout_rate=0.1)

        self.line1 = nn.Linear(entity_types_len * 2, entity_types_len * 2).to(0)
        self.line2 = nn.Linear(entity_types_len * 2, entity_types_len * 2).to(0)
        self.line3 = nn.Linear(entity_types_len * 2, entity_types_len).to(0)
        self.ln2 = nn.LayerNorm(entity_types_len, 1e-5, True)
        self.entity_types_len = entity_types_len
    def get_span_entity_hidden(self, batch1, batch2=None):
        
        if batch2 is not None:
            input_ids = torch.cat([batch1["input_ids"], batch2["input_ids"]], dim=0)
            attention_mask = torch.cat([batch1["input_mask"], batch2["input_mask"]], dim=0)
            token_type_ids = torch.cat([batch1["segment_ids"], batch2["segment_ids"]], dim=0)
        else:
            input_ids = batch1["input_ids"]
            attention_mask = batch1["input_mask"]
            token_type_ids = batch1["segment_ids"]

        max_len = (attention_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len].type(torch.int8)
        token_type_ids = token_type_ids[:, :max_len]
            
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_four_hidden_states = torch.cat(
            [hidden_state.unsqueeze(0) for hidden_state in output['hidden_states'][-4:]], 0)
        word_embeddings = torch.mean(last_four_hidden_states, 0)  # [num_sent, number_of_tokens, 768]
        span_hidden = self.dropout(output[0]) 
        sequence_output = span_hidden

        if self.project_type_embedding:
            entity_hidden = self.type_project(sequence_output)
        else:
            entity_hidden = sequence_output

        return span_hidden, entity_hidden, attention_mask, word_embeddings

    def get_filtered_entity_hidden(self, entity_hidden, types):

        B, M, D = entity_hidden.shape
        types = types.reshape(-1)
        mask = types > 0 
        entity_hidden = entity_hidden.reshape(-1, D)[mask]
        types = types[mask]

        return entity_hidden, types
    
    def get_entity_logits(self, entity_hidden, prototypes):

        if len(entity_hidden.shape) == 2 and len(prototypes.shape) == 3:
            entity_hidden_expand = entity_hidden.unsqueeze(0).unsqueeze(-2)
            support_prototypes_expand = prototypes.unsqueeze(1)

        if len(entity_hidden.shape) == 3 and len(prototypes.shape) == 3:
            entity_hidden_expand = entity_hidden.unsqueeze(0).unsqueeze(-2)
            support_prototypes_expand = prototypes.unsqueeze(1).unsqueeze(1)
        
        if len(entity_hidden.shape) == 3 and len(prototypes.shape) == 2:
            entity_hidden_expand = entity_hidden.unsqueeze(2)
            support_prototypes_expand = prototypes.unsqueeze(0).unsqueeze(0)

        type_logits = (entity_hidden_expand * support_prototypes_expand).sum(-1)

        return type_logits

    def calculate_span_loss(self, span_logits, labels):

        C = span_logits.shape[-1]
        label_mask = labels >= 0
        active_loss = label_mask.reshape(-1) == 1
        active_logits = span_logits.reshape(-1, C)[active_loss]
        active_labels = labels.reshape(-1)[active_loss]
        base_loss = self.span_loss(active_logits, active_labels, reduction="none")
        span_loss = torch.mean(base_loss)

        return span_loss
    
    def calculate_entity_loss(self, type_logits, types):

        if len(type_logits.shape) - len(types.shape) > 1:
            types = types.unsqueeze(0).expand(type_logits.shape[0], -1, -1)

        T = type_logits.shape[-1]
        active_types = types.reshape(-1)>0
        active_labels = types.reshape(-1)[active_types] - 1
        active_logits = type_logits.reshape(-1, T)[active_types]
        loss = self.type_loss(active_logits, active_labels, reduction="mean")

        return loss

    def calculate_label_loss_with_probs(self, label_probs, labels):

        if len(label_probs.shape)>3:
            labels = labels.unsqueeze(0).expand(label_probs.shape[0], -1, -1)
        
        C = label_probs.shape[-1]
        active_loss = labels.reshape(-1) >=0 
        active_probs = label_probs.reshape(-1, C)[active_loss]
        active_labels = labels.reshape(-1)[active_loss]
        # calculate cross_entropy loss
        mask = torch.arange(C, device=label_probs.device)[None, :] == active_labels[:, None]
        mask_probs = active_probs.masked_select(mask)
        label_loss = -1 * torch.mean(torch.log(mask_probs + 1e-5))

        return label_loss

    def get_class_information(self, data):
        class_pro_dict = {}
        id_freq = {}
        _, _, _, word_embeddings = self.get_span_entity_hidden(data, None)
        max_len = (data["input_mask"] != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
        types = data["type_ids"][:, : max_len]
        B, M = types.shape
        types = types.reshape(-1)
        emb = word_embeddings.view(B * M, -1)
        for i in range(len(data["types_map"])):
            id_freq[i] = 0
            class_pro_dict[i] = torch.zeros(emb.shape[-1]).to(types.device)
        for i in range(B * M):
            if types[i] >= 0:
                id_freq[types[i].item()] += 1
                class_pro_dict[types[i].item()] += emb[i]

        class_pro_information = torch.zeros(size=[len(class_pro_dict), emb.shape[-1]]).to(types.device)

        for i in range(len(class_pro_dict)):
            if id_freq[i] == 0:
                class_pro_information[i, :] = 0
            else:
                class_pro_information[i, :] = class_pro_dict[i] / id_freq[i]

        return class_pro_information

    def __dist_JS__(self, xs, ys, xu, yu):
        KL = 0.5 * (
            (ys / xs + xs / ys + (xu - yu) ** 2 * (1 / xs + 1 / ys)).sum(dim=-1)
            - self.dim * 2
        )
        return KL

    def __batch_dist_JS__(
        self,
        su: torch.Tensor,
        ss: torch.Tensor,
        qu: torch.Tensor,
        qs: torch.Tensor,
    ):
        return self.__dist_JS__(
            ss.unsqueeze(0).unsqueeze(1), qs.unsqueeze(2), su.unsqueeze(0).unsqueeze(1), qu.unsqueeze(2)
        )

    def compute_NCA_loss_by_dist(self, dists, tag):
        losses = []
        for label in range(torch.max(tag) + 1):
            Xp = (tag == label).sum()
            if Xp.item():
                A = dists[:, label][tag == label].sum()
                B = dists[:, label].sum()
                ce = -torch.log(A / B)
                losses.append(torch.where(torch.isfinite(ce), ce, torch.full_like(ce, 0)))

        loss = 0 if len(losses) == 0 else torch.stack(losses).mean()
        del losses
        return loss

    def loss(
        self, 
        support, 
        query,
        org_support,
        entity_types, 
        calculate_loss: bool=True, 
        gamma: float=0.4, 
        top_k: int=1, 
        lamda: float=1e-3, 
    ):
        # B x M x D
        span_hidden, entity_hidden, attention_mask, word_embeddings = self.get_span_entity_hidden(support, query)
        bsz = span_hidden.shape[0]
        self.class_information = self.get_class_information(org_support)
        #_, agu_entity_hidden, _ = self.get_span_entity_hidden(agu_support)
        ss, qs = self.elu1(self.fs(F.relu(self.class_information))) + 1 + 1e-6, self.elu2(
            self.fs(F.relu(word_embeddings))) + 1 + 1e-6
        su, qu = self.fu(F.relu(self.class_information)), self.fu(F.relu(word_embeddings))
        result = self.__batch_dist_JS__(su, ss, qu, qs).detach()

        #span, entity (support_len + quert_len, maxlen, hidden)
        # obtain labels, tags and types
        B, M, D = entity_hidden.shape
        labels = torch.cat([support["label_ids"], query["label_ids"]], dim=0)[:, :M]
        tags = torch.cat([support["tag_ids"], query["tag_ids"]], dim=0)[:, :M]
        types = torch.cat([support["type_ids"], query["type_ids"]], dim=0)[:, :M]
        types_list = support["types"]

        # split suppport set representation and query set representation
        num_supports = support["input_ids"].shape[0]
        support_indices = torch.arange(num_supports).to(span_hidden.device).long()
        query_indices = torch.arange(num_supports, entity_hidden.shape[0]).to(span_hidden.device).long()

        support_entity_hidden_filter, support_types = self.get_filtered_entity_hidden(entity_hidden[support_indices], types[support_indices])
        query_entity_hidden_filter, query_types = self.get_filtered_entity_hidden(entity_hidden[query_indices], types[query_indices])
        #agu_entity_hidden_filter, agu_support_types = self.get_filtered_entity_hidden(agu_entity_hidden, agu_support["type_ids"][:, :agu_entity_hidden.shape[1]])


        # obtain prototypes
        # mu_prior_list, logstd_prior_list = [], []
        mu_posterior_list, logstd_posterior_list = [], []

        for typ in range(1, len(types_list)+1):

            support_hiddens = support_entity_hidden_filter[support_types == typ]
            query_hiddens = query_entity_hidden_filter[query_types == typ]
            # agu_support_hiddens = agu_entity_hidden_filter[agu_support_types == typ]

            support_hiddens = None if len(support_hiddens)==0 else support_hiddens
            query_hiddens = None if len(query_hiddens) == 0 else query_hiddens
            # agu_support_hiddens = None if len(agu_support_hiddens) == 0 else agu_support_hiddens


            # mu_prior, logstd_prior = self.inference_block(support_hiddens, agu_support_hiddens, gamma, entity_hidden.device)
            mu_posterior, logstd_posterior = self.inference_block(support_hiddens, query_hiddens, gamma, entity_hidden.device)

            # mu_prior_list.append(mu_prior)
            # logstd_prior_list.append(logstd_prior)
            mu_posterior_list.append(mu_posterior)
            logstd_posterior_list.append(logstd_posterior)

        # infer posterior distribution of prototypes
        prototype_mu_posterior = torch.cat(mu_posterior_list, dim=0)
        prototype_logstd_posterior = torch.cat(logstd_posterior_list, dim=0)

        # concat
        prototypes = prototype_mu_posterior.unsqueeze(0) + torch.randn(self.sample_size, len(types_list), entity_hidden.shape[-1]).to(prototype_mu_posterior.device) * torch.exp(prototype_logstd_posterior.unsqueeze(0))

        # span logits
        span_logits = self.classifier(span_hidden) # B x M x C

        span_log = torch.zeros(result.size(0), result.size(1), 2).to(0)
        span_log[:, :, 0] = result[:, :, 0]
        values, _ = torch.topk(result[:, :, 1:], k=1, dim=-1, largest=False)
        span_log[:, :, 1] = values.squeeze(2)
        tem_logits = span_logits.clone()
        new_logits = self.line6(self.line5(tem_logits) * self.activate2(self.line4(span_log)))
        span_logits += new_logits

        # entity type logits
        type_logits = self.get_entity_logits(entity_hidden, prototypes) # B x M x C
        k = type_logits.shape[-1]
        padding = self.entity_types_len - k
        if padding > 0:
            zeros = torch.zeros(type_logits.size(0), type_logits.size(1), type_logits.size(2), padding).to(prototype_mu_posterior.device)
            enti_types = torch.cat((type_logits, zeros), dim=-1)
            zeros = torch.zeros(result.size(0), result.size(1), padding).to(prototype_mu_posterior.device)
            dis = torch.cat((result, zeros), dim=-1)[:, :, 1:]
        else:
            enti_types = type_logits
            dis = result[:, :, 1:]

        new_e_types = torch.cat((enti_types, dis.unsqueeze(0).repeat(self.sample_size, 1, 1, 1)), dim=-1)
        new_e_types = self.line3(self.line1(new_e_types) * self.activate(self.line2(new_e_types)))

        type_logits += new_e_types[:, :, :, : k]

        span_loss = None
        type_loss = None
        loss = None

        if calculate_loss:

            C = span_logits.shape[-1]
            span_logits = span_logits.unsqueeze(0).expand(type_logits.shape[0], -1, -1, -1)
            tags = tags.unsqueeze(0).expand(type_logits.shape[0], -1, -1, -1)

            span_loss = self.calculate_span_loss(span_logits, tags)
            span_probs = F.softmax(span_logits, dim=-1) # S x B x M x C

            type_loss = self.calculate_entity_loss(type_logits, types)
            type_probs = F.softmax(type_logits, dim=-1) # S x B x M x T

            active_loss = attention_mask.view(-1) == 1
            tag = types.reshape(-1)[active_loss]
            dis = torch.exp(-result)
            ac_result = dis.view(-1, dis.shape[-1])[active_loss]
            pro_loss = self.compute_NCA_loss_by_dist(ac_result, tag) / bsz

            label_probs_list = [span_probs[:, :, :, 0].unsqueeze(-1)]
            for i in range(1, C):
                label_probs_list.append(span_probs[:, :, :, i].unsqueeze(-1) * type_probs)
            label_probs = torch.cat(label_probs_list, dim=-1) # S x B x M x 1+(C-1)*T
            label_loss = self.calculate_label_loss_with_probs(label_probs, labels)


            loss = span_loss + type_loss + label_loss + 0.5 * pro_loss

        return loss, span_loss, type_loss, span_logits, torch.mean(type_logits, dim=0), torch.mean(label_probs)


    def loss_finetune(
        self, 
        support,
        org_support,
        entity_types, 
        calculate_loss: bool=True, 
        gamma: float=0.4, 
        top_k: int=1, 
        lamda: float=1e-3, 
        update_prototypes: bool=True, 
    ):

        span_hidden, entity_hidden, attention_mask, word_embeddings = self.get_span_entity_hidden(support, None)
        bsz = span_hidden.shape[0]
        self.class_information = self.get_class_information(org_support)
        # _, agu_entity_hidden, _ = self.get_span_entity_hidden(agu_support)
        ss, qs = self.elu1(self.fs(F.relu(self.class_information))) + 1 + 1e-6, self.elu2(
            self.fs(F.relu(word_embeddings))) + 1 + 1e-6
        su, qu = self.fu(F.relu(self.class_information)), self.fu(F.relu(word_embeddings))
        result = self.__batch_dist_JS__(su, ss, qu, qs).detach()
        # obtain labels, tags and types
        B, M, D = entity_hidden.shape
        labels = support["label_ids"][:, :M]
        tags = support["tag_ids"][:, :M]
        types = support["type_ids"][:, :M]
        types_list = support["types"]
        support_entity_hidden_filter, support_types = self.get_filtered_entity_hidden(entity_hidden, types)

        mu_prior_list, logstd_prior_list = [], []

        for typ in range(1, len(types_list)+1):

            support_hiddens = support_entity_hidden_filter[support_types == typ]
            support_hiddens = None if len(support_hiddens)==0 else support_hiddens

            mu_prior, logstd_prior = self.inference_block(support_hiddens, None, gamma, entity_hidden.device)
            mu_prior_list.append(mu_prior)
            logstd_prior_list.append(logstd_prior)

        # obtain prior distribution of prototypes
        prototype_mu_prior = torch.cat(mu_prior_list, dim=0)
        prototype_logstd_prior = torch.cat(logstd_prior_list, dim=0)

        if update_prototypes:
            self.prototypes = prototype_mu_prior

        # sample prototypes
        prototypes = prototype_mu_prior.unsqueeze(0) + torch.randn(self.sample_size, len(types_list),
                                                                   entity_hidden.shape[-1]).to(prototype_mu_prior.device) * torch.exp(prototype_logstd_prior.unsqueeze(0))

        # span logits
        span_logits = self.classifier(span_hidden)
        span_log = torch.zeros(result.size(0), result.size(1), 2).to(0)
        span_log[:, :, 0] = result[:, :, 0]
        values, _ = torch.topk(result[:, :, 1:], k=1, dim=-1, largest=False)
        span_log[:, :, 1] = values.squeeze(2)
        tem_logits = span_logits.clone()
        new_logits = self.line6(self.line5(tem_logits) * self.activate2(self.line4(span_log)))

        span_logits += new_logits
        # entity type logits
        type_logits = self.get_entity_logits(entity_hidden, prototypes)
        k = type_logits.shape[-1]
        padding = self.entity_types_len - k
        if padding > 0:
            zeros = torch.zeros(type_logits.size(0), type_logits.size(1), type_logits.size(2), padding).to(
                prototype_mu_prior.device)
            enti_types = torch.cat((type_logits, zeros), dim=-1)
            zeros = torch.zeros(result.size(0), result.size(1), padding).to(prototype_mu_prior.device)
            dis = torch.cat((result, zeros), dim=-1)[:, :, 1:]
        else:
            enti_types = type_logits
            dis = result[:, :, 1:]

        new_e_types = torch.cat((enti_types, dis.unsqueeze(0).repeat(self.sample_size, 1, 1, 1)), dim=-1)
        new_e_types = self.line3(self.line1(new_e_types) * self.activate(self.line2(new_e_types)))

        type_logits += new_e_types[:, :, :, : k]

        C = span_logits.shape[-1]
        span_logits = span_logits.unsqueeze(0).expand(type_logits.shape[0], -1, -1, -1)
        tags = tags.unsqueeze(0).expand(type_logits.shape[0], -1, -1, -1)
        span_loss = self.calculate_span_loss(span_logits, tags)
        span_probs = F.softmax(span_logits, dim=-1)

        type_loss = self.calculate_entity_loss(type_logits, types)
        type_probs = F.softmax(type_logits, dim=-1) # S x B x M x T

        active_loss = attention_mask.view(-1) == 1
        tag = types.reshape(-1)[active_loss]
        dis = torch.exp(-result)
        ac_result = dis.view(-1, dis.shape[-1])[active_loss]
        pro_loss = self.compute_NCA_loss_by_dist(ac_result, tag) / bsz

        label_probs_list = [span_probs[:, :, :, 0].unsqueeze(-1)]
        for i in range(1, C):
            label_probs_list.append(span_probs[:, :, :, i].unsqueeze(-1) * type_probs)
        label_probs = torch.cat(label_probs_list, dim=-1) # S x B x M x 1+(C-1)*T
        label_loss = self.calculate_label_loss_with_probs(label_probs, labels)


        loss = span_loss + type_loss + label_loss + 0.5 * pro_loss

        return loss, span_loss, type_loss, span_logits, torch.mean(type_logits, dim=0), torch.mean(label_probs)


    def predict(self, batch):
        
        span_hidden, entity_hidden, attention_mask, word_embeddings = self.get_span_entity_hidden(batch, None)

        ss, qs = self.elu1(self.fs(F.relu(self.class_information))) + 1 + 1e-6, self.elu2(
            self.fs(F.relu(word_embeddings))) + 1 + 1e-6
        su, qu = self.fu(F.relu(self.class_information)), self.fu(F.relu(word_embeddings))
        result = self.__batch_dist_JS__(su, ss, qu, qs).detach()
        
        span_logits = self.classifier(span_hidden) # B M C
        span_log = torch.zeros(result.size(0), result.size(1), 2).to(0)
        span_log[:, :, 0] = result[:, :, 0]
        values, _ = torch.topk(result[:, :, 1:], k=1, dim=-1, largest=False)
        span_log[:, :, 1] = values.squeeze(2)
        tem_logits = span_logits.clone()
        new_logits = self.line6(self.line5(tem_logits) * self.activate2(self.line4(span_log)))

        span_logits += new_logits

        type_logits = self.get_entity_logits(entity_hidden, self.prototypes) # B x M x P
        k = type_logits.shape[-1]
        padding = self.entity_types_len - k
        if padding > 0:
            zeros = torch.zeros(type_logits.size(0), type_logits.size(1), padding).to(0)
            enti_types = torch.cat((type_logits, zeros), dim=-1)
            zeros = torch.zeros(result.size(0), result.size(1), padding).to(0)
            dis = torch.cat((result, zeros), dim=-1)[:, :, 1:]
        else:
            enti_types = type_logits
            dis = result[:, :, 1:]

        new_e_types = torch.cat((enti_types, dis), dim=-1)
        new_e_types = self.line3(self.line1(new_e_types) * self.activate(self.line2(new_e_types)))

        type_logits += new_e_types[:, :, : k]

        B, M, C = span_logits.shape

        span_probs = F.softmax(span_logits, dim=-1)
        type_probs = F.softmax(type_logits, dim=-1)
        label_probs_list = [span_probs[:, :, 0].unsqueeze(-1)]
        for i in range(1, C):
            label_probs_list.append(span_probs[:, :, i].unsqueeze(-1) * type_probs)
        label_probs = torch.cat(label_probs_list, dim=-1)

        return span_logits, type_logits, label_probs


class Learner(nn.Module):
    
    ignore_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    pad_token_label_id = -100

    def __init__(self, bert_model, freeze_layer, logger, lr, warmup_prop, max_train_steps, entity_len, model_dir="", cache_dir="", gpu_no=0, args=None):

        super(Learner, self).__init__()

        self.entity_len = entity_len
        self.lr = lr
        self.warmup_prop = warmup_prop
        self.max_train_steps = max_train_steps
        self.bert_model = bert_model
        self.entity_types = args.entity_types
        self.is_debug = args.debug
        self.model_dir = model_dir
        self.args = args
        self.freeze_layer = freeze_layer
        self.choice_number = args.choice_number
        num_tags = self.entity_types.num_tags
        if model_dir == "":
            logger.info("********** Loading pre-trained model **********")
            cache_dir = cache_dir if cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
            self.model = ProtoBertForTokenClassification.from_pretrained(bert_model, \
                cache_dir=cache_dir, num_labels=num_tags, output_hidden_states=True)
            self.model.set_config(
                args.project_type_embedding, 
                args.type_embedding_size, 
                sample_size=self.args.sample_size,
                entity_types_len=self.entity_len
            )
            self.model.to(args.device)
            self.layer_set()

        else:
            logger.info("********** Loading saved model **********")
            self.load_model(self.model_dir, entitylen=self.entity_len)

    def layer_set(self):

        no_grad_param_names = ["embeddings", "pooler"] + ["layer.{}.".format(i) for i in range(self.freeze_layer)]
        logger.info("The frozen parameters are:")
        for name, param in self.model.named_parameters():
            if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                param.requires_grad = False
                logger.info("  {}".format(name))

        self.opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(self.max_train_steps * self.warmup_prop),
            num_training_steps=self.max_train_steps,
        )

    def get_optimizer_grouped_parameters(self):

        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params

    def load_weights(self, names, params):

        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)

    def load_gradients(self, names, grads):

        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is None:
                continue
            model_params[n].grad.data.add_(g.data)  # accumulate

    def get_learning_rate(self, lr, progress, warmup, schedule="linear"):
        
        if schedule == "linear":
            if progress < warmup:
                lr *= progress / warmup
            else:
                lr *= max((progress - 1.0) / (warmup - 1.0), 0.0)
        return lr

    def forward(self, batch_query, batch_support, progress, corpus):

        span_losses, type_losses, losses = [], [], []
        task_num = len(batch_query)
        
        self.model.train()
        for task_id in range(task_num):
            support = self.preprocessdata(batch_support[task_id], corpus)
            loss, span_loss, type_loss, span_logits, type_logits, label_probs = self.model.loss(
                support = support,
                query = batch_query[task_id],
                org_support= batch_support[task_id],
                entity_types=self.entity_types,
                calculate_loss=True,
                gamma=self.args.gamma,
                top_k=self.args.top_k,
                lamda=self.args.lamda,
            )

            loss.backward()
            span_losses.append(span_loss.item())
            type_losses.append(type_loss.item())
            losses.append(loss.item())

        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()

        return (
            np.mean(span_losses) if span_losses else 0,
            np.mean(type_losses) if type_losses else 0,
            np.mean(losses) if losses else 0,
        )

    def write_result(self, words, y_true, y_pred, tmp_fn):
        assert len(y_pred) == len(y_true)
        with open(tmp_fn, "w", encoding="utf-8") as fw:
            for i, sent in enumerate(y_true):
                for j, word in enumerate(sent):
                    fw.write("{} {} {}\n".format(words[i][j], word, y_pred[i][j]))
            fw.write("\n")

    def finetune(self, data_support, lr_curr, finetune_steps, corpus, no_grad: bool = False):

        finetune_opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr)

        self.model.train()
        for i in range(finetune_steps):

            finetune_opt.param_groups[0]["lr"] = lr_curr
            finetune_opt.param_groups[1]["lr"] = lr_curr

            finetune_opt.zero_grad()

            support = self.preprocessdata(data_support, corpus)
            loss, span_loss, type_loss, _, _, _ = self.model.loss_finetune(
                support=support,
                org_support=data_support,
                entity_types=self.entity_types,
                calculate_loss=True,
                gamma=self.args.gamma,
                top_k=self.args.top_k,
                lamda=self.args.lamda
            )

            if no_grad:
                continue

            loss.backward()
            finetune_opt.step()
        
        return loss.item()

    def preprocessdata_prompt(self, support, corpus):
        data = deepcopy(support)
        self.model.train()
        # 添加提示学习
        label_word_freq = corpus.label_word_freq
        label_word_label = corpus.label_word_label
        label_word_tag = corpus.label_word_tag
        label_word_type = corpus.label_word_type
        tokenizer = corpus.tokenizer
        type_dict = corpus.entity_types.types_dict
        types = data["types"]
        le = data["input_ids"].shape[0]

        label_word_li = []
        for i in range(len(self.entity_types.types_list)):
            label_word_li.append([])
            if len(label_word_freq[i]) != 0:
                for label_word in label_word_freq[i]:
                    label_word_li[i].append(list(label_word))

        max_len = (data["input_mask"] != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1

        for type in types:
            if max_len + len(label_word_li[type_dict[type]][0]) + 5 >= 128: break
            max_len += len(label_word_li[type_dict[type]][0]) + 5
        for i in range(data["input_ids"].shape[0]):
            l = 0
            for j in data["input_mask"][i]:
                if j == 0: break
                l += 1
            for type in types:
                prompt = ['is', type]
                pr = []
                for p in prompt:
                    word = tokenizer.tokenize(p)
                    if len(word) == 0:
                        continue
                    pr.extend(word)
                tokens = tokenizer.convert_tokens_to_ids(pr)
                new_tokens = [102]
                new_tokens += label_word_li[type_dict[type]][0] + tokens
                if l + len(new_tokens) >= min(128, max_len): break
                j = len(new_tokens)
                data["input_ids"][i][l: l + j] = torch.tensor(new_tokens)
                data["input_mask"][i][l: l + j] = 1
                data["segment_ids"][i][l: l + j] = 1
                data["label_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                data["tag_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                data["type_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                l += len(new_tokens)
        return data

    def preprocessdata(self, support, corpus):
        data = deepcopy(support)
        self.model.train()
        label_word_freq = corpus.label_word_freq
        label_word_label = corpus.label_word_label
        label_word_tag = corpus.label_word_tag
        label_word_type = corpus.label_word_type
        tokenizer = corpus.tokenizer
        type_dict = corpus.entity_types.types_dict
        types = data["types"]
        le = data["input_ids"].shape[0]
        input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tag_ids_list, type_ids_list = [], [], [], [], [], []
        label_word_li = []
        for i in range(len(self.entity_types.types_list)):
            label_word_li.append([])
            if len(label_word_freq[i]) != 0:
                for label_word in label_word_freq[i]:
                    label_word_li[i].append(list(label_word))
        for i in range(le):
            entities = data["entities"][i]
            entities_len = len(entities)
            if not entities is None:
                for en in entities:
                    if en[0] > en[1]:
                        logger.info(0)
            for en in range(entities_len):
                entitie = data["entities"][i][en]

                replace_indices = np.random.binomial(1, 0.5, entities_len).astype(bool)
                replace_indices[en] = entities_len == 1
                input_ids_list.append(deepcopy(data["input_ids"][i]))
                input_mask_list.append(deepcopy(data["input_mask"][i]))
                segment_ids_list.append(deepcopy(data["segment_ids"][i]))
                label_ids_list.append(deepcopy(data["label_ids"][i]))
                tag_ids_list.append(deepcopy(data["tag_ids"][i]))
                type_ids_list.append(deepcopy(data["type_ids"][i]))
                data["entities"].append(deepcopy(data["entities"][i]))
                for re in range(entities_len):
                    l = 0
                    for j in input_mask_list[-1]:
                        if j == 0: break
                        l += 1
                    if not replace_indices[re]: continue
                    choice_number = np.random.randint(0, self.choice_number)
                    begin = data["entities"][-1][re][0]
                    end = data["entities"][-1][re][1]
                    la = data["entities"][-1][re][2]
                    if choice_number >= len(label_word_li[la]):
                        choice_number = np.random.randint(0, len(label_word_li[la]))
                    choice_label_word = label_word_li[la][choice_number]
                    choice_label_word_len = len(choice_label_word)
                    x = abs(choice_label_word_len - end + begin - 1)
                    if x + l >= 128: continue
                    if choice_label_word_len == end - begin + 1:
                        for ll in range(len(choice_label_word)):
                            input_ids_list[-1][begin + ll] = choice_label_word[ll]
                        label_ids_list[-1][begin: end + 1] = torch.tensor([data["labels_map"][label] if label != -100 else label for label in label_word_label[la][tuple(choice_label_word)]])
                        tag_ids_list[-1][begin: end + 1] = torch.tensor(label_word_tag[la][tuple(choice_label_word)])
                        type_ids_list[-1][begin: end + 1] = torch.tensor([data["types_map"][ty] if ty != -100 else ty for ty in label_word_type[la][tuple(choice_label_word)]])
                    elif choice_label_word_len > end - begin + 1:
                        input_ids_list[-1][begin + x: l + x] = deepcopy(input_ids_list[-1][begin: l])
                        input_mask_list[-1][begin + x : l + x] = deepcopy(input_mask_list[-1][begin: l])
                        segment_ids_list[-1][begin + x: l + x] = deepcopy(segment_ids_list[-1][begin: l])
                        label_ids_list[-1][begin + x: l + x] = deepcopy(label_ids_list[-1][begin: l])
                        tag_ids_list[-1][begin + x: l + x] = deepcopy(tag_ids_list[-1][begin: l])
                        type_ids_list[-1][begin + x: l + x] = deepcopy(type_ids_list[-1][begin: l])
                        input_mask_list[-1][begin: begin + choice_label_word_len] = torch.ones(choice_label_word_len).to(device=self.args.gpu_device)
                        for ll in range(len(choice_label_word)):
                            input_ids_list[-1][begin + ll] = choice_label_word[ll]
                        label_ids_list[-1][begin: end + x + 1] = torch.tensor([data["labels_map"][label] if label != -100 else label for label in label_word_label[la][tuple(choice_label_word)]])
                        tag_ids_list[-1][begin: end + x + 1] = torch.tensor(label_word_tag[la][tuple(choice_label_word)])
                        type_ids_list[-1][begin: end + x + 1] = torch.tensor([data["types_map"][ty] if ty != -100 else ty for ty in label_word_type[la][tuple(choice_label_word)]])
                        data["entities"][-1][re] = (begin, end + x, la)
                        for rr in range(re + 1, len(data["entities"][-1])):
                            bee = data["entities"][-1][rr][0]
                            endd = data["entities"][-1][rr][1]
                            enen = data["entities"][-1][rr][2]
                            data["entities"][-1][rr] = (bee + x, endd + x, enen)
                    else:
                        input_ids_list[-1][begin: l] = deepcopy(input_ids_list[-1][begin + x: l + x])
                        input_mask_list[-1][begin: l] = deepcopy(input_mask_list[-1][begin + x: l + x])
                        segment_ids_list[-1][begin: l] = deepcopy(segment_ids_list[-1][begin + x: l + x])
                        label_ids_list[-1][begin: l] = deepcopy(label_ids_list[-1][begin + x: l + x])
                        tag_ids_list[-1][begin: l] = deepcopy(tag_ids_list[-1][begin + x: l + x])
                        type_ids_list[-1][begin: l] = deepcopy(type_ids_list[-1][begin + x: l + x])

                        for ll in range(len(choice_label_word)):
                            input_ids_list[-1][begin + ll] = choice_label_word[ll]
                        label_ids_list[-1][begin: begin + choice_label_word_len] = torch.tensor([data["labels_map"][label] if label != -100 else label for label in label_word_label[la][tuple(choice_label_word)]])
                        tag_ids_list[-1][begin: begin + choice_label_word_len] = torch.tensor(label_word_tag[la][tuple(choice_label_word)])
                        type_ids_list[-1][begin: begin + choice_label_word_len] = torch.tensor([data["types_map"][ty] if ty != -100 else ty for ty in label_word_type[la][tuple(choice_label_word)]])
                        data["entities"][-1][re] = (begin, begin + choice_label_word_len - 1, la)
                        for rr in range(re + 1, len(data["entities"][-1])):
                            bee = data["entities"][-1][rr][0]
                            endd = data["entities"][-1][rr][1]
                            enen = data["entities"][-1][rr][2]
                            data["entities"][-1][rr] = (bee - x, endd - x, enen)

                entitie = data["entities"][-1][en]
                b = entitie[0]
                e = entitie[1]
                prompt = ['is', self.entity_types.types_list[entities[en][2]]]
                pr = []
                for p in prompt:
                    word = tokenizer.tokenize(p)
                    if len(word) == 0:
                        continue
                    pr.extend(word)
                tokens = tokenizer.convert_tokens_to_ids(pr)
                new_tokens = []
                new_tokens += input_ids_list[-1][b: e + 1].cpu().tolist() + tokens
                l = 0
                for j in input_mask_list[-1]:
                    if j == 0: break
                    l += 1
                if l + len(new_tokens) >= 128: break
                j = len(new_tokens)
                input_ids_list[-1][l: l + j] = torch.tensor(new_tokens)
                input_mask_list[-1][l: l + j] = 1
                segment_ids_list[-1][l: l + j] = 1
                label_ids_list[-1][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                tag_ids_list[-1][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                type_ids_list[-1][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index

                if en == entities_len - 1:
                    entitie = data["entities"][i][en]
                    b = entitie[0]
                    e = entitie[1]
                    prompt = ['is', self.entity_types.types_list[entities[en][2]]]
                    pr = []
                    for p in prompt:
                        word = tokenizer.tokenize(p)
                        if len(word) == 0:
                            continue
                        pr.extend(word)
                    tokens = tokenizer.convert_tokens_to_ids(pr)
                    new_tokens = []
                    new_tokens += list(data["input_ids"][i][b: e + 1]) + tokens
                    l = 0
                    for j in data["input_mask"][i]:
                        if j == 0: break
                        l += 1
                    if l + len(new_tokens) >= 128: break
                    j = len(new_tokens)
                    data["input_ids"][i][l: l + j] = torch.tensor(new_tokens)
                    data["input_mask"][i][l: l + j] = 1
                    data["segment_ids"][i][l: l + j] = 1
                    data["label_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                    data["tag_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                    data["type_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index

        data["input_ids"] = torch.cat(
            (data["input_ids"], torch.tensor([ii.tolist() for ii in input_ids_list], device=self.args.gpu_device)),
            dim=0)
        data["input_mask"] = torch.cat(
            (data["input_mask"], torch.tensor([ii.tolist() for ii in input_mask_list], device=self.args.gpu_device)),
            dim=0)
        data["segment_ids"] = torch.cat(
            (data["segment_ids"], torch.tensor([ii.tolist() for ii in segment_ids_list], device=self.args.gpu_device)),
            dim=0)
        data["label_ids"] = torch.cat(
            (data["label_ids"], torch.tensor([ii.tolist() for ii in label_ids_list], device=self.args.gpu_device)),
            dim=0)
        data["tag_ids"] = torch.cat(
            (data["tag_ids"], torch.tensor([ii.tolist() for ii in tag_ids_list], device=self.args.gpu_device)), dim=0)
        data["type_ids"] = torch.cat(
            (data["type_ids"], torch.tensor([ii.tolist() for ii in type_ids_list], device=self.args.gpu_device)),
            dim=0)
        max_len = (data["input_mask"] != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1

        for type in types:
            if max_len + len(label_word_li[type_dict[type]][0]) + 5 >= 128: break
            max_len += len(label_word_li[type_dict[type]][0]) + 5
        for i in range(data["input_ids"].shape[0]):
            l = 0
            for j in data["input_mask"][i]:
                if j == 0: break
                l += 1
            for type in types:
                prompt = ['is', type]
                pr = []
                for p in prompt:
                    word = tokenizer.tokenize(p)
                    if len(word) == 0:
                        continue
                    pr.extend(word)
                tokens = tokenizer.convert_tokens_to_ids(pr)
                new_tokens = [102]
                new_tokens += label_word_li[type_dict[type]][0] + tokens
                if l + len(new_tokens) >= min(128, max_len): break
                j = len(new_tokens)
                data["input_ids"][i][l: l + j] = torch.tensor(new_tokens)
                data["input_mask"][i][l: l + j] = 1
                data["segment_ids"][i][l: l + j] = 1
                data["label_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                data["tag_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                data["type_ids"][i][l: l + j] = torch.nn.CrossEntropyLoss().ignore_index
                l += len(new_tokens)

        del input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tag_ids_list, type_ids_list

        return data

    def batch_eval(self, data, use_viterbi=False, transition_matrix=None):

        N = data["input_ids"].shape[0]
        B = 32
        BATCH_KEY = ["input_ids", "input_mask", "segment_ids", "label_ids", "tag_ids", "type_ids"]

        num_types = len(data["types"])

        probs = []
        labels = []
        words = []

        if use_viterbi:
            pseudo_id2label = {i: i for i in range(self.entity_types.num_tags)}
            transition_matrix = transition_matrix.to(self.args.device)
            viterbi_decoder = ViterbiDecoder(pseudo_id2label, transition_matrix)
        else:
            viterbi_decoder = None
            
        for i in range((N - 1) // B + 1):
            tmp = {
                key: data[key][i * B : (i + 1) * B] for key in BATCH_KEY
            }

            tmp_span_logits, tmp_type_logits, tmp_label_probs = self.model.predict(batch=tmp)

            B, M, C = tmp_label_probs.shape
            if viterbi_decoder is None:
                probs.extend(tmp_label_probs.detach().cpu().numpy())
            else:

                log_span_probs = torch.log(F.softmax(tmp_span_logits, dim=-1) + 1e-6)
                mask = tmp["input_mask"][:, :M] 
                ids = tmp["tag_ids"][:, :M] 
                pred_tags = viterbi_decoder.forward(log_span_probs, mask, ids)
                tmp_span_probs = torch.zeros(tmp_span_logits.shape).to(self.args.device)
                
                for j in range(len(pred_tags)):
                    token_mask = ids[j] >= 0 
                    idx = 0
                    for k in range(len(token_mask)):
                        if token_mask[k]:
                            tmp_span_probs[j, k, pred_tags[j][idx]] = 1.0
                            idx += 1 

                tmp_type_probs = F.softmax(tmp_type_logits, dim=-1)
                tmp_probs_list = [tmp_span_probs[:, :, 0].unsqueeze(-1)]
                for j in range(1, self.entity_types.num_tags):
                    tmp_probs_list.append(tmp_span_probs[:, :, j].unsqueeze(-1) * tmp_type_probs)
                tmp_probs = torch.cat(tmp_probs_list, dim=-1)

                probs.extend(tmp_probs.detach().cpu().numpy())

            tmp_labels = tmp["label_ids"][:, :M]
            word_ids = tmp["input_ids"][:, :M]
            labels.extend(tmp_labels.detach().cpu().numpy())
            words.extend(word_ids.detach().cpu().numpy())

        return probs, labels, words
    
    def calculate_preds_scores(self, preds_all, labels_all):

        performance_dict = performance_measure(labels_all, preds_all)
        pred_sum, tp_sum, true_sum = filtered_tp_counts(labels_all, preds_all)

        preds_all = [[x.replace("E-", "I-").replace("S-", "B-") for x in preds_sent] for preds_sent in preds_all]
        labels_all =[[x.replace("E-", "I-").replace("S-", "B-") for x in labels_sent] for labels_sent in labels_all]

        results = {
            "precision": precision_score(labels_all, preds_all),
            "recall": recall_score(labels_all, preds_all),
            "f1": f1_score(labels_all, preds_all),
            "TP": performance_dict['TP'],
            "TN": performance_dict['TN'],
            "FP": performance_dict['FP'],
            "FN": performance_dict['FN'],
            "pred_sum": pred_sum,
            "tp_sum": tp_sum,
            "true_sum": true_sum
        }

        return results

    def get_most_frequent_type(self, types_list):

        types_dict = {}
        for t in types_list:
            types_dict[t] = types_dict.get(t, 0) + 1 
        result = "O"
        max_freq = -1 
        for t in types_dict.keys():
            if types_dict[t] <= max_freq: continue
            result = t 
            max_freq = types_dict[t]
        
        return result

    def post_process(self, label_list):

        type_list = [x.replace("B-", "").replace("I-", "").replace("E-", "").replace("S-", "") for x in label_list]
        span_list = []

        i = 0
        while(i < len(label_list)):
            if label_list[i][0] == 'B' or label_list[i][0] == "I":
                start_idx = i 
                i += 1 
                while(i < len(label_list) and not (label_list[i][0]=='O' or label_list[i][0] == 'E')):
                    i += 1 
                if i < len(label_list):
                    end_idx = i if label_list[i][0] == 'E' else i - 1 
                    span_list.append((start_idx, end_idx, self.get_most_frequent_type(type_list[start_idx: end_idx+1])))
                    i += 1 
                else:
                    span_list.append((start_idx, i-1, self.get_most_frequent_type(type_list[start_idx: i])))
            
            elif label_list[i][0] =='S' or label_list[i][0] == 'E':
                span_list.append((i, i, type_list[i]))
                i += 1 
            else:
                i += 1 

        result_label_list = ["O"] * len(label_list)
        for s, e, t in span_list:
            if s == e:
                result_label_list[s] = "S-" + t 
            else:
                result_label_list[s] = "B-" + t 
                result_label_list[e] = "E-" + t 
                for i in range(s+1, e):
                    result_label_list[i] = "I-" + t 
        
        return result_label_list

    def evaluate(self, corpus, logger, lr, steps, set_type):

        # set_type: "valid", "test"
        if self.is_debug:
            self.save_model(self.args.result_dir, "begin", self.args.max_seq_len, "all")

        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        t_tmp = time.time()
        labels_all, preds_all, words_all = [], [], []

        tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)
        transition_matrix = torch.zeros([self.entity_types.num_tags, self.entity_types.num_tags])
        if self.entity_types.num_tags == 3:
            transition_matrix[2][0] = -10000  # p(O -> I) = 0
        elif self.entity_types.num_tags == 5:
            for (i, j) in [(2, 0), (3, 0), (0, 1), (1, 1), (4, 1), (0, 2), (1, 2), (4, 2), (2, 3), (3, 3), (2, 4), (3, 4)]:
                transition_matrix[i][j] = -10000
        else:
            raise ValueError("Only support BIO and BIOES tagging scheme!")

        f1_score_per_task = []
        for item_id in trange(0, corpus.n_total, desc="Evaluate Steps: ", disable=False):
            
            eval_query, eval_support = corpus.get_batch_meta(batch_size=1, shuffle=False)
            self.finetune(eval_support[0], lr_curr=lr, finetune_steps=steps, corpus=corpus)
            reverse_labels_map = eval_support[0]["reverse_labels_map"]
            types_map = eval_support[0]["types_map"]

            self.model.eval()
            func = lambda x: self.entity_types.id2label[reverse_labels_map[x]]
            preds_per_task = []
            labels_per_task = []
            with torch.no_grad():
                probs, labels, words = self.batch_eval(eval_query[0], use_viterbi=True, transition_matrix=transition_matrix)
                preds = [np.argmax(prob, -1) for prob in probs]
                idx = 0
                for pred, label in zip(preds, labels):
                    mask = label >=0 
                    pred_labels = self.post_process(list(map(func, list(pred[mask]))))
                    true_labels = list(map(func, list(label[mask]))) 
                    preds_per_task.append(pred_labels)
                    labels_per_task.append(true_labels)
                    preds_all.append(pred_labels)
                    labels_all.append(true_labels)
                    words_all.append(
                        tokenizer.convert_ids_to_tokens(list(words[idx][mask]))
                    )
                    idx += 1

            f1_score_per_task.append(self.calculate_preds_scores(preds_per_task, labels_per_task)["f1"])
            self.load_weights(names, weights)

        store_dir = self.args.model_dir if self.args.model_dir else self.args.result_dir
        joblib.dump([labels_all, preds_all, words_all], "{}/{}_{}_preds.pkl".format(store_dir, "all", set_type))

        results = self.calculate_preds_scores(preds_all, labels_all)

        logger.info("************* Eval results: %s ******************", set_type)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("Average f1 score across tasks (reported): {:.7f}".format(np.mean(f1_score_per_task)))
        logger.info("*********************************************************")

        return results, preds_all

    def save_model(self, result_dir, fn_prefix, max_seq_len, mode: str = "all"):

        # Save a trained model and the associated configuration
        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)  # Only save the model it-self
        output_model_file = os.path.join(
            result_dir, "{}_{}_{}".format(fn_prefix, mode, WEIGHTS_NAME)
        )
        torch.save(model_to_save.state_dict(), output_model_file)

        # save config file
        output_config_file = os.path.join(result_dir, CONFIG_NAME)
        with open(output_config_file, "w", encoding="utf-8") as f:
            f.write(model_to_save.config.to_json_string())

        # save some configs
        model_config = {
            "bert_model": self.bert_model,
            "do_lower": False,
            "max_seq_length": max_seq_len
        }
        json.dump(
            model_config,
            open(
                os.path.join(result_dir, f"{mode}-model_config.json"),
                "w",
                encoding="utf-8",
            ),
        )

        joblib.dump(self.entity_types, os.path.join(result_dir, "type_embedding.pkl"))

    def load_model(self, model_dir: str = "", mode: str = "all", entitylen = None):

        if not model_dir:
            return

        logger.info(f"********** Loading saved model at: {model_dir} **********")
        output_model_file = os.path.join(model_dir, "en_{}_{}".format(mode, WEIGHTS_NAME))
        self.model = ProtoBertForTokenClassification.from_pretrained(self.bert_model, num_labels=self.entity_types.num_tags, output_hidden_states=True)
        self.model.set_config(
            self.args.project_type_embedding,
            self.args.type_embedding_size,
            sample_size=self.args.sample_size,
            entity_types_len=entitylen
        )
        self.model.to(self.args.device)
        self.model.load_state_dict(torch.load(output_model_file, map_location="cuda"))
        self.layer_set()


class ViterbiDecoder(object):

    def __init__(
        self,
        id2label,
        transition_matrix,
        ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
    ):
        self.id2label = id2label
        self.n_labels = len(id2label)
        self.transitions = transition_matrix
        self.ignore_token_label_id = ignore_token_label_id

    def forward(self, logprobs, attention_mask, label_ids):
        # probs: batch_size x max_seq_len x n_labels
        batch_size, max_seq_len, n_labels = logprobs.size()
        attention_mask = attention_mask[:, :max_seq_len]
        label_ids = label_ids[:, :max_seq_len]

        active_tokens = (attention_mask == 1) & (
            label_ids != self.ignore_token_label_id
        )
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][
                active_tokens[idx]
            ]  # seq_len(active) x n_labels

            back_pointers = []

            forward_var = logprob_i[0]  # n_labels

            for j in range(1, len(logprob_i)):  # for tag_feat in feat:
                next_label_var = forward_var + self.transitions  # n_labels x n_labels
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1)  # n_labels

                logp_j = logprob_i[j]  # n_labels
                forward_var = viterbivars_t + logp_j  # n_labels
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append(best_path)

        return label_seqs