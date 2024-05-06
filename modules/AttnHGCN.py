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

    def forward(self, user_emb, item_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        """
        Forward pass of the network, integrating item and user embeddings aggregation.
        """
        aggregated_item_emb = self.aggregate_items(item_emb, edge_index, edge_type)
        user_emb = self.aggregate_user_after_items(user_emb, aggregated_item_emb, inter_edge, inter_edge_w)
        return user_emb, aggregated_item_emb

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
