import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
from logging import getLogger


class AttnHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network enhanced with attention mechanism.
    """

    def __init__(self, channel, n_hops, n_users,
                 n_relations,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(torch.Tensor(n_relations - 1, channel))
        nn.init.xavier_uniform_(self.relation_emb)

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def compute_attention_weights(self, head_emb, rel_emb, tail_emb, edge_index):
        """
        Compute attention weights for edges in the graph.
        New: Attention mechanism using relation embeddings.
        """
        scores = torch.exp(torch.sum(head_emb * rel_emb * tail_emb, dim=1))
        normalized_scores = scatter_softmax(scores, edge_index[0])
        return normalized_scores

    def aggregate_items(self, entity_emb, edge_index, edge_type):
        """
        Multi-hop aggregation for items using attention.
        New: Attention-based aggregation logic.
        """
        for _ in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            entity_emb = F.normalize(self.dropout(entity_emb))  # Apply dropout and normalize
        return entity_emb

    def aggregate_user_after_items(self, user_emb, item_emb, inter_edge, inter_edge_w):
        """
        Aggregate user embeddings based on the previously aggregated item embeddings.
        """
        item_agg = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=self.n_users, dim=0)
        return F.normalize(user_agg)  # Normalize the aggregated user embeddings

    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def forward(self, user_emb, item_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        """
        Forward pass of the network, integrating item and user embeddings aggregation.
        """
        aggregated_item_emb = self.aggregate_items(item_emb, edge_index, edge_type)
        user_emb = self.aggregate_user_after_items(user_emb, aggregated_item_emb, inter_edge, inter_edge_w)
        return user_emb, aggregated_item_emb

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb

    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        """ Perform knowledge graph aggregation with attention """
        head, tail = edge_index
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        rel_emb = self.relation_emb[edge_type - 1]  # Adjust relation embedding indices

        # Compute attention weights
        attn_weights = self.compute_attention_weights(head_emb, rel_emb, tail_emb, edge_index)

        # Apply attention weights
        weighted_neigh_emb = attn_weights.unsqueeze(-1) * tail_emb
        entity_agg = scatter_sum(weighted_neigh_emb, head, dim=0, dim_size=entity_emb.size(0))
        return entity_agg

    def kg_agg_with_attention(self, entity_emb, edge_index, edge_type):
        """
        Knowledge graph aggregation with attention.
        New: Enhanced with relation-specific attention computation.
        """
        head, tail = edge_index
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        rel_emb = self.relation_emb[edge_type - 1]

        attn_scores = self.compute_attention_weights(head_emb, rel_emb, tail_emb, edge_index)
        weighted_tail_emb = attn_scores.unsqueeze(-1) * tail_emb
        entity_agg = scatter_sum(weighted_tail_emb, head, dim=0, dim_size=entity_emb.size(0))
        return entity_agg
    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        """
        Compute normalized attention scores for visualization or further analysis.
        """
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        logits = attn.mean(-1).detach()
        attn_scores = scatter_softmax(logits, head)
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        attn_scores *= norm
        if print:
            self.logger.info("edge_attn_score std: {}".format(attn_scores.std()))
        return attn_scores if not return_logits else (attn_scores, logits)