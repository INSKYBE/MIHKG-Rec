from rich.table import Table
from torch.nn import CosineSimilarity

from rich.console import Console
import numpy as np
import torch
import torch.nn as nn
from .AttnHGCN import AttnHGCN
from .contrast import Contrast
from logging import getLogger
import torch.nn.functional as F


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MIHKG(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat, hp_dict=None):
        super(MIHKG, self).__init__()
        self.args_config = args_config
        self.logger = getLogger()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.ablation = args_config.ab
        self.console = Console()
        self.mae_coef = args_config.mae_coef
        self.mae_msize = args_config.mae_msize
        self.cl_coef = args_config.cl_coef
        self.tau = args_config.cl_tau
        self.cl_drop = args_config.cl_drop_ratio

        self.samp_func = "torch"

        if args_config.dataset == 'last-fm':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.01
            self.tau = 1.0
            self.cl_drop = 0.5
        elif args_config.dataset == 'amazon-book':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.1
            self.cl_drop = 0.6
            self.samp_func = "np"
        elif args_config.dataset == 'alibaba':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.2
            self.cl_drop = 0.5

        # update hps
        if hp_dict is not None:
            for k, v in hp_dict.items():
                setattr(self, k, v)

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            adj_mat)

        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        # Initialize embeddings and model components
        self.initialize_embeddings()
        self.gcn = AttnHGCN(channel=self.emb_size,
                            n_hops=self.context_hops,
                            n_users=self.n_users,
                            n_relations=self.n_relations,
                            node_dropout_rate=self.node_dropout_rate,
                            mess_dropout_rate=self.mess_dropout_rate)
        self.mlp_se = MLP(self.emb_size, self.emb_size)
        self.mlp_gt = MLP(self.emb_size, self.emb_size)
        self.cos_sim = CosineSimilarity(dim=1)
        self.contrast_fn = Contrast(self.emb_size, tau=self.tau)

        self.lambda1 = 0.1  # Weight of the self-supervised loss
        self.lambda2 = 0.1  # Weight of the contrastive loss
        self.masking_percentage = 0.1  # Percentage of edges to mask

        # self.print_shapes()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        all_embed_init = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.all_embed = nn.Parameter(all_embed_init)

    def initialize_embeddings(self):
        # Initializer for embeddings
        initializer = torch.nn.init.xavier_uniform_
        self.all_embed = nn.Parameter(initializer(torch.empty(self.n_nodes, self.emb_size)))

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def compute_contrastive_loss(self, z_t, z_e):
        """
        Compute the contrastive loss using an improved InfoNCE loss.
        Ensure both tensors have the same size for comparison.
        """
        # Ensure both tensors are the same size
        min_size = min(z_t.size(0), z_e.size(0))
        z_t = z_t[:min_size]
        z_e = z_e[:min_size]

        # Compute cosine similarities
        pos_sim = self.cos_sim(z_t, z_e) / self.tau

        # Generate negative samples
        idx = torch.randperm(min_size)
        neg_sim_t = self.cos_sim(z_t, z_e[idx]) / self.tau
        neg_sim_e = self.cos_sim(z_t[idx], z_e) / self.tau

        # Compute contrastive loss
        exp_pos = torch.exp(pos_sim)
        exp_neg_t = torch.exp(neg_sim_t)
        exp_neg_e = torch.exp(neg_sim_e)
        contrastive_loss = -torch.log(exp_pos / (exp_pos + exp_neg_t + exp_neg_e))
        return contrastive_loss.mean()

    def mask_edges(self, edge_index, num_edges_to_mask):
        """ Randomly mask some edges for the self-supervised learning task """
        all_indices = torch.randperm(edge_index.size(1))[:num_edges_to_mask]
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[all_indices] = False
        masked_edge_index = edge_index[:, all_indices]
        remaining_edge_index = edge_index[:, mask]
        return remaining_edge_index, masked_edge_index

    def reconstruct_edges(self, masked_edge_index, node_embeddings):
        """ Reconstruct masked edges """
        src, dst = masked_edge_index
        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]
        reconstructed_embeddings = torch.sigmoid((src_emb * dst_emb).sum(dim=1))
        return reconstructed_embeddings

    def forward(self, batch):
        user_indices = batch['users']
        pos_item_indices = batch['pos_items']
        neg_item_indices = batch['neg_items']

        user_emb = self.all_embed[:self.n_users]
        item_emb = self.all_embed[self.n_users:]

        # Perform multi-hop aggregation for the project
        aggregated_item_emb = self.gcn.aggregate_items(item_emb, self.edge_index, self.edge_type)
        # Aggregation-based project embedding aggregates user embedding
        aggregated_user_emb = self.gcn.aggregate_user_after_items(user_emb, aggregated_item_emb, self.inter_edge,
                                                                  self.inter_edge_w)

        # Apply MLPs
        z_se = self.mlp_se(aggregated_item_emb)
        z_gt = self.mlp_gt(aggregated_user_emb)

        # Calculate comparative learning loss
        contrastive_loss = self.compute_contrastive_loss(z_gt, z_se)

        # Mask edges and reconstruct them
        num_edges_to_mask = int(self.masking_percentage * self.edge_index.size(1))
        remaining_edge_index, masked_edge_index = self.mask_edges(self.edge_index, num_edges_to_mask)
        reconstructed_embeddings = self.reconstruct_edges(masked_edge_index, self.all_embed)
        true_values = torch.ones(masked_edge_index.size(1), device=self.device)
        mae_loss = F.binary_cross_entropy(reconstructed_embeddings, true_values)

        # Calculate BPR loss
        pos_scores = torch.sum(aggregated_user_emb[user_indices] * aggregated_item_emb[pos_item_indices], dim=1)
        neg_scores = torch.sum(aggregated_user_emb[user_indices] * aggregated_item_emb[neg_item_indices], dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization term (calculated once and reused)
        regularizer = (torch.norm(aggregated_user_emb[user_indices]) ** 2 + torch.norm(aggregated_item_emb) ** 2) / 2
        emb_loss = self.decay * regularizer / user_indices.size(0)

        # Total loss calculation
        total_loss = mf_loss + emb_loss + self.lambda1 * mae_loss + self.lambda2 * contrastive_loss

        loss_dict = {
            "rec_loss": mf_loss.item(),
            "reg_loss": emb_loss.item(),
            "mae_loss": mae_loss.item(),
            "contrast_loss": contrastive_loss.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, loss_dict

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.inter_edge,
                        self.inter_edge_w)[:2]

    def print_shapes(self):
        table = Table(title="Model Parameters and Hyperparameters")

        table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Ablation", str(self.ablation))
        table.add_row("tau", str(self.contrast_fn.tau))
        table.add_row("cL_drop", str(self.cl_drop))
        table.add_row("cl_coef", str(self.cl_coef))
        table.add_row("mae_coef", str(self.mae_coef))
        table.add_row("mae_msize", str(self.mae_msize))
        table.add_row("context_hops", str(self.context_hops))
        table.add_row("node_dropout", str(self.node_dropout))
        table.add_row("node_dropout_rate", f"{self.node_dropout_rate:.1f}")
        table.add_row("mess_dropout", str(self.mess_dropout))
        table.add_row("mess_dropout_rate", f"{self.mess_dropout_rate:.1f}")
        table.add_row("all_embed", str(self.all_embed.shape))
        table.add_row("interact_mat", str(self.inter_edge.shape))
        table.add_row("edge_index", str(self.edge_index.shape))
        table.add_row("edge_type", str(self.edge_type.shape))

        self.console.print(table)
