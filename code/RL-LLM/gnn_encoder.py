"""
GNN 图特征编码器

将 NetworkX 冲突图 G 转换为 PyTorch 张量格式，
通过图注意力网络 (GAT) 提取拓扑结构特征。

输出:
  - z_GNN: (batch_size, embed_dim) 图级特征向量
  - node_embeddings: (batch_size, max_nodes, embed_dim) 节点级特征 (用于 PPO 剪枝打分)
  - node_mask: (batch_size, max_nodes) 有效节点掩码

纯 PyTorch 实现，无需 torch-geometric 依赖。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import networkx as nx


# ============================================================
# 图特征提取辅助函数
# ============================================================

NODE_FEATURE_DIM = 21


def _numbers(node):
    return [list(map(int, re.findall(r'\d+', item))) for item in node]


def extract_node_features(G, Env, Agent_Data, Task_Time=None,
                          CurrTimeAge=1, SingleStep=10, TimeLimit=120):
    """
    从 NetworkX 图 G 中提取每个节点的初始特征向量。

    节点格式: (worker_str, car_str, uav_str, task_str, charging_str)
    其中各字段为 'None' 或带位置/索引的字符串。

    返回:
        node_features: np.ndarray, shape (num_nodes, NODE_FEATURE_DIM)
        node_to_idx: dict, 节点 -> 索引映射
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32), {}

    node_features = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 预计算度
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # 预计算每个节点的类型和邻居列表 (供 [3] [5] 使用)
    node_types = {}
    node_neighbors = {}
    for node in nodes:
        node_numbers = _numbers(node)
        worker = node_numbers[0]
        car = node_numbers[1]
        uav = node_numbers[2]
        task = node_numbers[3]
        charging = node_numbers[4]
        node_types[node] = _get_node_type(worker, car, uav, task, charging)
        node_neighbors[node] = list(G.neighbors(node))

    weights = np.asarray([float(G.nodes[node].get('weight', 0.0)) for node in nodes])
    weight_mean, weight_std = float(weights.mean()), float(weights.std())
    weight_std = max(weight_std, 1e-8)
    order = np.argsort(weights, kind='stable')
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    map_diagonal = max(math.hypot(Env.shape[1] - 1, Env.shape[2] - 1), 1.0)
    now = CurrTimeAge * SingleStep

    for i, node in enumerate(nodes):
        node_numbers = _numbers(node)

        worker = node_numbers[0]
        car = node_numbers[1]
        uav = node_numbers[2]
        task = node_numbers[3]
        charging = node_numbers[4]

        # [0:10] one-hot node type (permutation and scale invariant)
        node_type = node_types[node]
        node_features[i, node_type] = 1.0

        # [10:15] graph-relative utility and conflict features
        deg = degrees.get(node, 0)
        neighbors = node_neighbors[node]
        neighbor_weights = np.asarray([G.nodes[x].get('weight', 0.0) for x in neighbors], dtype=float)
        node_features[i, 10] = np.clip((weights[i] - weight_mean) / weight_std, -5, 5) / 5
        node_features[i, 11] = ranks[i]
        node_features[i, 12] = deg / max(max_degree, 1)
        if len(neighbor_weights):
            node_features[i, 13] = np.clip((neighbor_weights.mean() - weight_mean) / weight_std, -5, 5) / 5
            node_features[i, 14] = np.clip((neighbor_weights.max() - weight_mean) / weight_std, -5, 5) / 5

        # [15:21] normalized scheduling quantities used by creatWeight.
        destination = task or charging
        agent_fields = [field for field in (worker, car, uav) if field]
        distances, durations, end_times = [], [], []
        for field in agent_fields:
            idx, row, col = field[0], field[1], field[2]
            dist = math.hypot(destination[0] - row, destination[1] - col) if destination else 0.0
            speed = max(float(Agent_Data[idx, 6]), 1e-8)
            distances.append(dist)
            durations.append(dist / speed)
            end_times.append(float(Agent_Data[idx, 4]))
        task_cost = float(Task_Time[destination[0], destination[1]]) if (task and Task_Time is not None) else 0.0
        duration = (max(durations) if durations else 0.0) + task_cost
        node_features[i, 15] = (max(distances) if distances else 0.0) / map_diagonal
        node_features[i, 16] = min(duration / max(TimeLimit, 1), 2.0) / 2.0
        if uav:
            full_power = max(float(Agent_Data[uav[0], 8]), 1e-8)
            node_features[i, 17] = float(Agent_Data[uav[0], 7]) / full_power
            uav_dist = math.hypot(destination[0] - uav[1], destination[1] - uav[2]) if destination else 0.0
            node_features[i, 18] = min(uav_dist / full_power, 2.0) / 2.0
        node_features[i, 19] = min(task_cost / max(TimeLimit, 1), 1.0)
        slack = (min(end_times) - now - duration) / max(TimeLimit, 1) if end_times else 0.0
        node_features[i, 20] = (np.clip(slack, -1.0, 1.0) + 1.0) / 2.0

    return node_features, node_to_idx


def _get_node_type(worker, car, uav, task, charging):
    """根据 5 元组确定节点类型编码 (0-9)"""
    if worker and not car and uav and task and not charging:
        return 5  # W+U+T (任务三元组)
    if not worker and car and uav and not task and charging:
        return 8  # C+U+S (充电三元组)
    if worker and not car and not uav and task and not charging:
        return 3  # W+T (工人+任务)
    if not worker and not car and uav and task and not charging:
        return 4  # U+T (无人机+任务)
    if not worker and car and not uav and not task and charging:
        return 6  # C+S (补给车+充电点)
    if not worker and not car and uav and not task and charging:
        return 7  # U+S (无人机+充电点)
    if worker and not car and not uav and not task and not charging:
        return 0  # 空闲工人
    if not worker and car and not uav and not task and not charging:
        return 1  # 空闲补给车
    if not worker and not car and uav and not task and not charging:
        return 2  # 空闲无人机
    return 9  # 其他


def networkx_to_adjacency(G, node_to_idx):
    """
    将 NetworkX 图转换为邻接矩阵和节点特征批量。

    返回:
        adj: (N, N) 邻接矩阵 (带自环)
        node_feat: (N, 7) 节点特征
    """
    N = len(node_to_idx)
    if N == 0:
        return np.zeros((0, 0)), np.zeros((0, 7))

    adj = np.zeros((N, N))
    for node_i, idx_i in node_to_idx.items():
        for node_j in G.neighbors(node_i):
            idx_j = node_to_idx[node_j]
            adj[idx_i, idx_j] = 1.0
    # 加自环
    adj = adj + np.eye(N)
    return adj


# ============================================================
# GAT 层 (纯 PyTorch 实现)
# ============================================================

class GATLayer(nn.Module):
    """
    单层图注意力网络 (GAT)，多头注意力。

    Args:
        in_dim: 输入特征维度
        out_dim: 每个头的输出维度
        num_heads: 注意力头数
        dropout: dropout 概率
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # 线性变换 W (每个头独立)
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        # 注意力向量 a
        self.a = nn.Parameter(torch.randn(num_heads, 2 * out_dim) * 0.02)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x: (N, in_dim) 节点特征
            adj: (N, N) 邻接矩阵 (含自环, 0/1)
        Returns:
            h: (N, out_dim * num_heads) 输出特征
        """
        N = x.size(0)
        device = x.device

        # 线性变换: (N, out_dim * num_heads)
        Wh = self.W(x)  # (N, out_dim * H)
        Wh = Wh.view(N, self.num_heads, self.out_dim)  # (N, H, D')

        # 计算注意力分数
        # a1 和 a2 分别对应两个节点
        # Wh: (N, H, D'), a: (H, 2*D') -> (N, H, 1)
        a1 = (Wh * self.a[:, :self.out_dim].unsqueeze(0)).sum(dim=-1, keepdim=True)  # (N, H, 1)
        a2 = (Wh * self.a[:, self.out_dim:].unsqueeze(0)).sum(dim=-1, keepdim=True)  # (N, H, 1)

        # 注意力矩阵: e_ij = a1_i + a2_j
        e = a1.unsqueeze(1) + a2.unsqueeze(0)  # (N, N, H, 1)
        e = self.leaky_relu(e.squeeze(-1))  # (N, N, H)

        # 掩码: 只在邻居间计算注意力
        adj_expanded = adj.unsqueeze(-1).repeat(1, 1, self.num_heads)  # (N, N, H)
        e = e.masked_fill(adj_expanded == 0, float('-inf'))

        # Softmax 归一化
        alpha = F.softmax(e, dim=1)  # (N, N, H)
        alpha = self.dropout(alpha)

        # 加权聚合 (简单实现)
        # alpha: (N, N, H) -> (H, N, N)
        # Wh: (N, H, D') -> (H, N, D')
        alpha_t = alpha.permute(2, 0, 1)  # (H, N, N)
        Wh_t = Wh.permute(1, 0, 2)  # (H, N, D')
        h = torch.bmm(alpha_t, Wh_t)  # (H, N, D')
        h = h.permute(1, 0, 2).contiguous()  # (N, H, D')
        h = h.view(N, self.num_heads * self.out_dim)  # (N, H*D')

        return h


# ============================================================
# GNN 编码器主体
# ============================================================

class GNNEncoder(nn.Module):
    """
    图注意力网络编码器。

    架构:
      - GAT Layer 1: NODE_FEATURE_DIM -> 64, 4 heads
      - ELU + Dropout
      - GAT Layer 2: 256 -> 128, 1 head
      - 全局均值池化 -> z_GNN
    """

    def __init__(self, in_dim=7, hidden_dim=64, out_dim=128,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gat2 = GATLayer(hidden_dim * num_heads, out_dim, num_heads=1, dropout=dropout)

    def forward(self, x, adj):
        """
        Args:
            x: (N, in_dim) 节点特征
            adj: (N, N) 邻接矩阵 (含自环)
        Returns:
            z_GNN: (out_dim,) 图级特征向量
            node_emb: (N, out_dim) 节点级 embedding
        """
        if x.size(0) == 0:
            # 空图
            z = torch.zeros(self.out_dim, device=x.device)
            return z, x.new_zeros(0, self.out_dim)

        # 第一层 GAT
        h1 = self.gat1(x, adj)  # (N, hidden_dim * num_heads)
        h1 = self.elu(h1)
        h1 = self.dropout(h1)

        # 第二层 GAT
        h2 = self.gat2(h1, adj)  # (N, out_dim)

        # 全局均值池化
        z_GNN = h2.mean(dim=0)  # (out_dim,)

        return z_GNN, h2

    def batch_forward(self, node_features_list, adj_list):
        """
        批量前向传播 (处理一个 batch 的图)。

        Args:
            node_features_list: list of tensors, 每个 (N_i, NODE_FEATURE_DIM)
            adj_list: list of tensors, 每个 (N_i, N_i)

        Returns:
            z_GNN_batch: (batch_size, out_dim)
            node_emb_batch: (batch_size, max_nodes, out_dim)
            node_mask: (batch_size, max_nodes)
        """
        batch_size = len(node_features_list)
        if batch_size == 0:
            return (
                torch.zeros(0, self.out_dim),
                torch.zeros(0, 0, self.out_dim),
                torch.zeros(0, 0),
            )

        max_nodes = max(f.shape[0] if hasattr(f, 'shape') else len(f) for f in node_features_list) if node_features_list else 0
        if max_nodes == 0:
            max_nodes = 1

        device = next(self.parameters()).device

        z_list = []
        emb_padded = torch.zeros(batch_size, max_nodes, self.out_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, device=device)

        for b, (feat, adj) in enumerate(zip(node_features_list, adj_list)):
            feat_t = torch.tensor(feat, dtype=torch.float, device=device)
            adj_t = torch.tensor(adj, dtype=torch.float, device=device)
            n = feat_t.size(0)

            z, emb = self.forward(feat_t, adj_t)
            z_list.append(z)
            emb_padded[b, :n, :] = emb
            mask[b, :n] = 1.0

        z_GNN_batch = torch.stack(z_list, dim=0)  # (B, out_dim)

        return z_GNN_batch, emb_padded, mask


def create_gnn_encoder(in_dim=NODE_FEATURE_DIM, hidden_dim=64, out_dim=128, num_heads=4, dropout=0.1):
    """工厂函数: 创建 GNNEncoder 实例"""
    return GNNEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
