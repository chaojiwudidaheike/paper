"""
RL 剪枝环境

包装单步图剪枝决策的状态表示和奖励计算。
不存储状态，只提供状态编码和奖励计算的接口。
"""

import numpy as np
import torch


class PruningEnv:
    """
    图剪枝 RL 环境。

    在 CpOL-AsHs 的每个时间步中：
    1. build_graph() 生成完整图 G
    2. 本环境提取状态特征 (LLM + GNN)
    3. PPO 智能体决定剪枝
    4. 剪枝后的图用于 MWIS 求解
    5. 本环境根据求解结果计算奖励
    """

    def __init__(self, llm_encoder, gnn_encoder, device='cpu'):
        """
        Args:
            llm_encoder: LLMEncoder 实例
            gnn_encoder: GNNEncoder 实例
            device: torch 设备
        """
        self.llm_encoder = llm_encoder
        self.gnn_encoder = gnn_encoder
        self.device = device

        # 奖励权重
        self.alpha = 1.0     # 能效权重
        self.beta = 0.01     # 响应时间权重 (负向，越小越好)
        self.gamma_prune = 0.1  # 剪枝率权重

        # 记录上一个时间步的统计 (用于增量奖励)
        self.prev_completed_tasks = 0
        self.prev_total_uav_dist = 0.0

    def extract_state(self, G, Agent_Data, Env, Task_Time,
                      centerRow, centerLin, Center_Scope,
                      CurrTimeAge, SingleStep, TimeLimit, grad=False):
        """
        从当前调度状态提取特征向量。

        返回:
            z_llm:    (1, E) LLM 文本特征
            z_gnn:    (1, E) GNN 图特征
            node_emb: (1, max_N, E) 节点级 embedding
            node_mask:(1, max_N) 有效节点掩码
            node_to_idx: dict 节点到索引的映射 (用于应用剪枝)
        """
        from llm_encoder import build_text_prompt
        from gnn_encoder import extract_node_features, networkx_to_adjacency

        # 1. LLM 特征
        text = build_text_prompt(
            Agent_Data, Env, centerRow, centerLin, Center_Scope,
            CurrTimeAge, SingleStep, TimeLimit, Task_Time,
        )
        node_count = len(G)
        max_edges = node_count * (node_count - 1) / 2
        density = G.number_of_edges() / max(max_edges, 1)
        text += f" graph_size={np.log1p(node_count) / 10:.4f} graph_density={density:.4f}"
        if grad:
            z_llm = self.llm_encoder([text]).to(self.device)
        else:
            with torch.no_grad():
                z_llm = self.llm_encoder([text]).to(self.device)

        # 2. GNN 特征
        node_feat, node_to_idx = extract_node_features(
            G, Env, Agent_Data, Task_Time, CurrTimeAge, SingleStep, TimeLimit,
        )
        adj = networkx_to_adjacency(G, node_to_idx)

        # 为 batch_forward 准备数据
        if grad:
            z_gnn, node_emb, node_mask = self.gnn_encoder.batch_forward([node_feat], [adj])
        else:
            with torch.no_grad():
                z_gnn, node_emb, node_mask = self.gnn_encoder.batch_forward([node_feat], [adj])

        z_gnn = z_gnn.to(self.device)
        node_emb = node_emb.to(self.device)
        node_mask = node_mask.to(self.device)

        return z_llm, z_gnn, node_emb, node_mask, node_to_idx

    def safe_keep_mask(self, G, keep_probs, node_mask, node_to_idx,
                       keep_ratio=0.5):
        """Return a score-based mask with feasible-set and coverage anchors."""
        probs = keep_probs.detach().cpu().numpy().squeeze(0)
        nodes = list(node_to_idx)
        budget = max(1, int(np.ceil(len(nodes) * keep_ratio)))
        mandatory = set()

        # A weighted greedy independent set is a deployable quality floor.
        greedy = set()
        for node in sorted(nodes, key=lambda n: G.nodes[n].get('weight', 0.0), reverse=True):
            if all(not G.has_edge(node, chosen) for chosen in greedy):
                greedy.add(node)
        mandatory.update(greedy)

        # Retain the best incident candidate for every target and every agent.
        groups = {}
        for node, idx in node_to_idx.items():
            for field_idx, field in enumerate(node):
                if field != 'None':
                    prefix = 'target' if field_idx in (3, 4) else 'agent'
                    identifier = field.split('|', 1)[0] if prefix == 'agent' else field
                    groups.setdefault((prefix, field_idx, identifier), []).append((probs[idx], node))
        for candidates in groups.values():
            mandatory.add(max(candidates, key=lambda item: item[0])[1])

        keep_count = max(budget, len(mandatory))
        ranked = sorted(nodes, key=lambda node: probs[node_to_idx[node]], reverse=True)
        selected = set(mandatory)
        for node in ranked:
            if len(selected) >= keep_count:
                break
            selected.add(node)
        mask = torch.zeros_like(node_mask)
        for node in selected:
            mask[0, node_to_idx[node]] = 1.0
        mask *= node_mask
        return mask, {
            'budget': budget,
            'mandatory': len(mandatory),
            'kept': int(mask.sum().item()),
            'budget_exceeded': len(mandatory) > budget,
            'greedy_nodes': greedy,
        }

    @staticmethod
    def validate_pruning(G_pruned, safety_info):
        return (len(G_pruned) > 0 and
                safety_info['greedy_nodes'].issubset(set(G_pruned.nodes())))

    def apply_pruning(self, G, keep_mask, node_to_idx):
        """
        根据剪枝掩码修改 NetworkX 图 G。

        Args:
            G: NetworkX Graph (会被原地修改! 请在外部传入副本)
            keep_mask: (1, max_N) 二值保留掩码
            node_to_idx: dict 节点 -> 索引

        Returns:
            G: 剪枝后的图
            n_pruned: 被剪掉的节点数
        """
        keep_mask_np = keep_mask.detach().cpu().numpy().squeeze(0)  # (max_N,)
        idx_to_node = {v: k for k, v in node_to_idx.items()}

        nodes_to_remove = []
        for node, idx in node_to_idx.items():
            if idx < len(keep_mask_np) and keep_mask_np[idx] < 0.5:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            if node in G:
                G.remove_node(node)

        return G, len(nodes_to_remove)

    def compute_reward(self, completed_tasks, total_uav_dist_task,
                       atrt, n_pruned, n_total_nodes):
        """
        根据剪枝后的 MWIS 求解结果计算奖励。

        Args:
            completed_tasks: 累计完成任务数
            total_uav_dist_task: 累计 UAV 任务飞行距离
            atrt: 平均任务响应时间
            n_pruned: 被剪枝的节点数
            n_total_nodes: 原始图节点数

        Returns:
            reward: float
        """
        # 能效
        EE = completed_tasks / max(total_uav_dist_task, 1e-6)

        # 响应时间惩罚
        atrt_penalty = -atrt if atrt > 0 else 0.0

        # 剪枝率奖励
        prune_ratio = n_pruned / max(n_total_nodes, 1)
        prune_bonus = prune_ratio

        reward = (
            self.alpha * EE +
            self.beta * atrt_penalty +
            self.gamma_prune * prune_bonus
        )

        return reward

    def reset_stats(self):
        """重置累计统计 (每个 episode 开始时调用)"""
        self.prev_completed_tasks = 0
        self.prev_total_uav_dist = 0.0
