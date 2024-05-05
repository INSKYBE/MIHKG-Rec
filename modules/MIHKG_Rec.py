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


def _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
    _, least_attn_edge_id = torch.topk(-edge_attn_score,
                                       int((1 - keep_rate) * edge_attn_score.shape[0]), sorted=False)
    cl_kg_mask = torch.ones_like(edge_attn_score).bool()
    cl_kg_mask[least_attn_edge_id] = False
    cl_kg_edge = edge_index[:, cl_kg_mask]
    cl_kg_type = edge_type[cl_kg_mask]
    return cl_kg_edge, cl_kg_type


def _adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func="torch"):
    inter_attn_prob = item_attn_mean[inter_edge[1]]
    # add gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
    """ prob based drop """
    inter_attn_prob = inter_attn_prob + noise
    inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

    if samp_func == "np":
        # we observed abnormal behavior of torch.multinomial on mind
        sampled_edge_idx = np.random.choice(np.arange(inter_edge_w.shape[0]),
                                            size=int(keep_rate * inter_edge_w.shape[0]), replace=False,
                                            p=inter_attn_prob.cpu().numpy())
    else:
        sampled_edge_idx = torch.multinomial(inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False)

    return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx] / keep_rate


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v


def batched_matmul(batches, user_emb, item_emb, tau, device):
    sim_scores = []
    batch_user_size = user_emb.shape[0] // batches  # 用户分批大小
    batch_item_size = item_emb.shape[0] // batches  # 项目分批大小

    for i in range(batches):
        for j in range(batches):
            start_user = i * batch_user_size
            end_user = min(user_emb.shape[0], (i + 1) * batch_user_size)
            start_item = j * batch_item_size
            end_item = min(item_emb.shape[0], (j + 1) * batch_item_size)

            batch_scores = torch.matmul(user_emb[start_user:end_user], item_emb[start_item:end_item].t()) / tau
            sim_scores.append(F.softmax(batch_scores, dim=1))

    # 将分批结果拼接回完整矩阵
    weights = torch.cat([torch.cat(sim_scores[i::batches], dim=1) for i in range(batches)], dim=0)
    return weights


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
        elif args_config.dataset == 'mind-f':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.1
            self.cl_drop = 0.6
            self.samp_func = "np"
        elif args_config.dataset == 'alibaba-fashion':
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

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def contrastive_loss(self, embeddings1, embeddings2):
        # 这里是计算对比学习损失的一个简单例子
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        return F.mse_loss(norm_embeddings1, norm_embeddings2)

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

        # 为项目执行多跳聚合
        aggregated_item_emb = self.gcn.aggregate_items(item_emb, self.edge_index, self.edge_type)
        # 基于聚合的项目嵌入聚合用户嵌入
        aggregated_user_emb = self.gcn.aggregate_user_after_items(user_emb, aggregated_item_emb, self.inter_edge,
                                                                  self.inter_edge_w)

        # Apply MLPs
        z_se = self.mlp_se(aggregated_item_emb)
        z_gt = self.mlp_gt(aggregated_user_emb)

        # 计算对比学习损失
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

    def calc_topk_attn_edge(self, entity_emb, edge_index, edge_type, k):
        edge_attn_score = self.gcn.norm_attn_computer(
            entity_emb, edge_index, edge_type, return_logits=True)
        positive_mask = edge_attn_score > 0
        edge_attn_score = edge_attn_score[positive_mask]
        edge_index = edge_index[:, positive_mask]
        edge_type = edge_type[positive_mask]
        topk_values, topk_indices = torch.topk(
            edge_attn_score, k, sorted=False)
        return edge_index[:, topk_indices], edge_type[topk_indices]

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.inter_edge,
                        self.inter_edge_w)[:2]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # @TimeCounter.count_time(warmup_interval=4)
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        if torch.isnan(mf_loss):
            raise ValueError("nan mf_loss")

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

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

    def generate_kg_drop(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        edge_index, edge_type = _edge_sampling(
            self.edge_index, self.edge_type, self.kg_drop_test_keep_rate)
        return self.gcn(user_emb,
                        item_emb,
                        edge_index,
                        edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
                        mess_dropout=False)[:2]

    def generate_global_attn_score(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        edge_attn_score = self.gcn.norm_attn_computer(
            item_emb, self.edge_index, self.edge_type)

        return edge_attn_score, self.edge_index, self.edge_type
