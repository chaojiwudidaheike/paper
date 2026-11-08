"""
PPO 剪枝智能体

包含:
  - 门控融合层: 将 LLM 特征和 GNN 特征融合为 h_fused
  - Actor 网络: 对每个节点输出保留概率
  - Critic 网络: 输出状态价值 V(s)
  - PPO 训练逻辑

门控融合公式:
  g = sigmoid(W_g · [proj(z_LLM), z_GNN])
  h_fused = g * proj(z_LLM) + (1 - g) * z_GNN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ============================================================
# 门控融合层
# ============================================================

class GatedFusion(nn.Module):
    """
    LLM-GNN 门控融合模块。

    将 LLM 文本特征 z_LLM 和 GNN 图特征 z_GNN 融合为 h_fused。
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        # LLM 特征投影
        self.llm_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 门控网络: [proj(z_LLM), z_GNN] -> gate value
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_llm, z_gnn):
        """
        Args:
            z_llm: (batch_size, embed_dim) LLM 文本特征
            z_gnn:  (batch_size, embed_dim) GNN 图特征
        Returns:
            h_fused: (batch_size, embed_dim) 融合特征
            gate:    (batch_size, 1) 门控值 (用于监控)
        """
        # 投影 LLM 特征
        z_llm_proj = self.llm_proj(z_llm)  # (B, E)

        # 计算门控值
        gate_input = torch.cat([z_llm_proj, z_gnn], dim=-1)  # (B, 2E)
        gate = self.gate_net(gate_input)  # (B, 1)

        # 门控融合
        h_fused = gate * z_llm_proj + (1 - gate) * z_gnn  # (B, E)

        return h_fused, gate


# ============================================================
# Actor 网络 (策略网络)
# ============================================================

class Actor(nn.Module):
    """
    策略网络: 对每个图节点输出保留概率。

    Input:  h_fused (B, embed_dim) + node_emb (B, max_N, embed_dim)
    Output: keep_probs (B, max_N) — 每个节点的保留概率
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        # 将 h_fused 扩展为与节点同维的特征
        self.fuse_proj = nn.Linear(embed_dim, embed_dim)

        # 节点级评分 MLP
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, h_fused, node_emb, node_mask):
        """
        Args:
            h_fused:   (B, embed_dim) 融合状态特征
            node_emb:  (B, max_N, embed_dim) 各节点 GNN embedding
            node_mask: (B, max_N) 有效节点掩码 (1=有效)
        Returns:
            keep_probs: (B, max_N) 每个节点的保留概率 [0, 1]
        """
        B, max_N, E = node_emb.shape

        # 将 h_fused 扩展到每个节点
        h_fused_proj = self.fuse_proj(h_fused)  # (B, E)
        h_fused_expanded = h_fused_proj.unsqueeze(1).expand(B, max_N, E)  # (B, max_N, E)

        # 拼接: [h_fused, node_emb]
        node_input = torch.cat([h_fused_expanded, node_emb], dim=-1)  # (B, max_N, 2E)

        # 节点评分
        scores = self.scorer(node_input).squeeze(-1)  # (B, max_N)

        # Sigmoid -> 保留概率
        keep_probs = torch.sigmoid(scores)

        # 无效节点概率置 0
        keep_probs = keep_probs * node_mask

        return keep_probs

    def get_action(self, h_fused, node_emb, node_mask, deterministic=False,
                   min_keep_ratio=0.1):
        """
        根据当前策略采样剪枝动作。

        Args:
            h_fused, node_emb, node_mask: 同 forward
            deterministic: True 时使用确定性策略 — 按 keep_probs 从大到小排序，
                           保留期望数量 ceil(sum(keep_probs)) 个节点（有下限保护）
            min_keep_ratio: 确定性模式下最低保留比例 (避免将图完全剪空)
        Returns:
            keep_mask: (B, max_N) 二值掩码 1=保留, 0=剪枝
            keep_probs: (B, max_N) 保留概率
            log_probs:  (B, max_N) 对数概率 (用于 PPO)
        """
        keep_probs = self.forward(h_fused, node_emb, node_mask)

        if deterministic:
            # 自适应 top-K: 保留数量 = max(ceil(sum(probs)), floor(n_valid * min_keep_ratio))
            B, max_N = node_mask.shape
            keep_mask = torch.zeros_like(keep_probs)
            for b in range(B):
                n_valid = int(node_mask[b].sum().item())
                if n_valid == 0:
                    continue
                probs_b = keep_probs[b, :n_valid]  # 只取有效节点
                # 期望保留数 = sum(probs) 但不少于 min_keep_ratio
                keep_cnt = max(1, int(torch.ceil(probs_b.sum()).item()),
                               int(n_valid * min_keep_ratio))
                keep_cnt = min(keep_cnt, n_valid)
                # 按概率排序取 top-K
                _, top_indices = torch.topk(probs_b, keep_cnt)
                keep_mask[b, top_indices] = 1.0
            keep_mask = keep_mask * node_mask
        else:
            # 采样 Bernoulli
            keep_mask = torch.bernoulli(keep_probs) * node_mask

        # 对数概率: log(p) if keep=1, log(1-p) if keep=0
        log_probs = keep_mask * torch.log(keep_probs + 1e-8) + \
                    (1 - keep_mask) * torch.log(1 - keep_probs + 1e-8)
        # 无效节点 log_prob 置 0
        log_probs = log_probs * node_mask

        return keep_mask, keep_probs, log_probs

    def evaluate_actions(self, h_fused, node_emb, node_mask, keep_mask):
        """
        评估已执行动作的对数概率和熵 (用于 PPO 更新)。

        Returns:
            log_probs: (B, max_N) 动作对数概率
            entropy:   (B,) 平均熵
        """
        keep_probs = self.forward(h_fused, node_emb, node_mask)

        # 对数概率
        log_probs = keep_mask * torch.log(keep_probs + 1e-8) + \
                    (1 - keep_mask) * torch.log(1 - keep_probs + 1e-8)
        log_probs = log_probs * node_mask

        # 熵: -[p·log(p) + (1-p)·log(1-p)]
        entropy_per_node = -(keep_probs * torch.log(keep_probs + 1e-8) +
                             (1 - keep_probs) * torch.log(1 - keep_probs + 1e-8))
        entropy_per_node = entropy_per_node * node_mask
        # 每个样本的平均熵 (在有效节点上取平均)
        n_valid = node_mask.sum(dim=1).clamp(min=1)
        entropy = entropy_per_node.sum(dim=1) / n_valid  # (B,)

        return log_probs, entropy


# ============================================================
# Critic 网络 (价值网络)
# ============================================================

class Critic(nn.Module):
    """
    价值网络: 评估当前融合状态的价值 V(s)。
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, h_fused):
        """
        Args:
            h_fused: (B, embed_dim) 融合状态
        Returns:
            value: (B, 1) 状态价值
        """
        return self.net(h_fused)


# ============================================================
# PPO 智能体
# ============================================================

class PruningPPO:
    """
    PPO 剪枝智能体，包含门控融合、Actor、Critic 以及训练逻辑。
    """

    def __init__(self, embed_dim=128, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=10, device='cpu'):
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.device = device

        # 网络模块
        self.fusion = GatedFusion(embed_dim=embed_dim).to(device)
        self.actor = Actor(embed_dim=embed_dim).to(device)
        self.critic = Critic(embed_dim=embed_dim).to(device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.fusion.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=lr,
        )

        # 经验缓冲区
        self.buffer = deque(maxlen=50000)

        # 计数器
        self.train_step = 0

    def fuse(self, z_llm, z_gnn):
        """门控融合: z_LLM + z_GNN -> h_fused"""
        return self.fusion(z_llm, z_gnn)

    @torch.no_grad()
    def act(self, z_llm, z_gnn, node_emb, node_mask, deterministic=False,
            min_keep_ratio=0.1):
        """
        执行一次剪枝决策。

        Args:
            z_llm:    (B, E) LLM 特征
            z_gnn:    (B, E) GNN 图特征
            node_emb: (B, max_N, E) 节点 embedding
            node_mask:(B, max_N) 有效节点掩码
            deterministic: 是否确定性决策
            min_keep_ratio: 确定性模式下最低保留比例

        Returns:
            keep_mask:  (B, max_N) 剪枝掩码
            keep_probs: (B, max_N) 保留概率
            h_fused:    (B, E) 融合特征 (用于 critic)
            gate:       (B, 1) 门控值
        """
        h_fused, gate = self.fusion(z_llm, z_gnn)
        keep_mask, keep_probs, log_probs = self.actor.get_action(
            h_fused, node_emb, node_mask,
            deterministic=deterministic, min_keep_ratio=min_keep_ratio,
        )
        return keep_mask, keep_probs, log_probs, h_fused, gate

    @torch.no_grad()
    def get_value(self, z_llm, z_gnn):
        """获取当前状态的价值估计"""
        h_fused, _ = self.fusion(z_llm, z_gnn)
        return self.critic(h_fused)

    def store_transition(self, z_llm, z_gnn, node_emb, node_mask,
                         keep_mask, old_log_probs, reward, done):
        """存储一个时间步的经验到缓冲区"""
        def _to_numpy(t):
            if t is None:
                return None
            if isinstance(t, np.ndarray):
                return t
            return t.detach().cpu().numpy()

        self.buffer.append({
            'z_llm': _to_numpy(z_llm),
            'z_gnn': _to_numpy(z_gnn),
            'node_emb': _to_numpy(node_emb),
            'node_mask': _to_numpy(node_mask),
            'keep_mask': _to_numpy(keep_mask),
            'old_log_probs': _to_numpy(old_log_probs),
            'reward': reward,
            'done': done,
        })

    def collect_episode_data(self):
        """
        从缓冲区收集最近一个 episode 的数据用于 PPO 更新。

        由于不同 timestep 的图节点数不同 (max_N 变化)，
        需要对 node_emb, node_mask, keep_mask 做 padding 到统一维度。

        返回格式化的 numpy 数组。
        """
        if len(self.buffer) == 0:
            return None

        data = {
            'z_llm': [],
            'z_gnn': [],
            'node_emb': [],
            'node_mask': [],
            'keep_mask': [],
            'old_log_probs': [],
            'rewards': [],
            'dones': [],
        }

        # 只保留有效数据 (跳过空图 None 条目)
        valid_items = [it for it in self.buffer if it['z_llm'] is not None]
        if not valid_items:
            return None

        # 第一遍: 找最大节点数
        max_N = 1
        for item in valid_items:
            n = item['node_emb'].shape[1]
            if n > max_N:
                max_N = n

        E = self.embed_dim

        # 第二遍: 收集并 padding, squeeze 掉 batch 维度使最终形状为 (T, ...)
        for item in valid_items:
            # z_llm/z_gnn: stored as (1, E) -> squeeze to (E,) for stacking to (T, E)
            data['z_llm'].append(item['z_llm'].squeeze(0))
            data['z_gnn'].append(item['z_gnn'].squeeze(0))

            n = item['node_emb'].shape[1]
            # node_emb: (1, N, E) -> squeeze + pad -> (max_N, E) for stacking to (T, max_N, E)
            if n < max_N:
                padded_emb = np.zeros((max_N, E), dtype=np.float32)
                if n > 0:
                    padded_emb[:n, :] = item['node_emb'].squeeze(0)
                data['node_emb'].append(padded_emb)
                # node_mask: (1, N) -> squeeze + pad -> (max_N,)
                padded_mask = np.zeros((max_N,), dtype=np.float32)
                if n > 0:
                    padded_mask[:n] = item['node_mask'].squeeze(0)
                data['node_mask'].append(padded_mask)
                # keep_mask: (1, N) -> squeeze + pad -> (max_N,)
                padded_keep = np.zeros((max_N,), dtype=np.float32)
                if n > 0:
                    padded_keep[:n] = item['keep_mask'].squeeze(0)
                data['keep_mask'].append(padded_keep)
                padded_log_probs = np.zeros((max_N,), dtype=np.float32)
                if n > 0:
                    padded_log_probs[:n] = item['old_log_probs'].squeeze(0)
                data['old_log_probs'].append(padded_log_probs)
            else:
                data['node_emb'].append(item['node_emb'].squeeze(0))
                data['node_mask'].append(item['node_mask'].squeeze(0))
                data['keep_mask'].append(item['keep_mask'].squeeze(0))
                data['old_log_probs'].append(item['old_log_probs'].squeeze(0))

            data['rewards'].append(item['reward'])
            data['dones'].append(item['done'])

        return {k: np.array(v) for k, v in data.items()}

    def update(self):
        """
        PPO 更新: 用缓冲区中的经验训练网络。

        返回:
            avg_loss: 平均损失
            avg_gate: 平均门控值
        """
        if len(self.buffer) < 2:
            return 0.0, 0.0

        data = self.collect_episode_data()
        T = len(data['rewards'])

        # 转换为 tensor
        z_llm = torch.tensor(data['z_llm'], dtype=torch.float, device=self.device)  # (T, E)
        z_gnn = torch.tensor(data['z_gnn'], dtype=torch.float, device=self.device)  # (T, E)
        node_emb = torch.tensor(data['node_emb'], dtype=torch.float, device=self.device)  # (T, max_N, E)
        node_mask = torch.tensor(data['node_mask'], dtype=torch.float, device=self.device)  # (T, max_N)
        keep_mask = torch.tensor(data['keep_mask'], dtype=torch.float, device=self.device)  # (T, max_N)
        old_log_probs = torch.tensor(data['old_log_probs'], dtype=torch.float, device=self.device)
        rewards = torch.tensor(data['rewards'], dtype=torch.float, device=self.device)  # (T,)
        dones = torch.tensor(data['dones'], dtype=torch.float, device=self.device)  # (T,)

        # 1. 计算 GAE 优势估计
        with torch.no_grad():
            h_fused, _ = self.fusion(z_llm, z_gnn)  # (T, E)
            values = self.critic(h_fused).squeeze(-1)  # (T,)

            advantages = torch.zeros(T, device=self.device)
            returns = torch.zeros(T, device=self.device)

            gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0  # episode 结束后价值为 0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # 2. PPO 更新 (多个 epoch)
        total_loss = 0.0
        total_gate = 0.0

        for _ in range(self.update_epochs):
            h_fused, gate = self.fusion(z_llm, z_gnn)  # (T, E)

            # Actor 损失
            new_log_probs, entropy = self.actor.evaluate_actions(
                h_fused, node_emb, node_mask, keep_mask
            )
            # Node-wise clipped objective avoids graph-size-dependent joint ratios.
            ratio = torch.exp((new_log_probs - old_log_probs).clamp(-20, 20))
            advantage_nodes = advantages.unsqueeze(1)
            surr1 = ratio * advantage_nodes
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantage_nodes
            actor_loss = -(torch.min(surr1, surr2) * node_mask).sum() / node_mask.sum().clamp(min=1)

            # Critic 损失
            values_pred = self.critic(h_fused).squeeze(-1)  # (T,)
            critic_loss = F.mse_loss(values_pred, returns)

            # 熵正则化
            entropy_loss = -entropy.mean() * 0.01

            # 总损失
            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            total_gate += gate.mean().item()

        self.train_step += 1
        avg_loss = total_loss / self.update_epochs
        avg_gate = total_gate / self.update_epochs

        return avg_loss, avg_gate

    def clear_buffer(self):
        """清空经验缓冲区"""
        self.buffer.clear()

    def save_model(self, path_prefix, llm_encoder=None, gnn_encoder=None,
                   extra_state=None):
        """Save the complete policy; partial checkpoints are not deployable."""
        torch.save(self.fusion.state_dict(), f"{path_prefix}_fusion")
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic")
        if llm_encoder is not None:
            torch.save(llm_encoder.state_dict(), f"{path_prefix}_llm")
        if gnn_encoder is not None:
            torch.save(gnn_encoder.state_dict(), f"{path_prefix}_gnn")
        torch.save({'optimizer': self.optimizer.state_dict(),
                    'train_step': self.train_step,
                    'extra_state': extra_state or {}}, f"{path_prefix}_training")
        print(f"[PPO] 模型已保存至 {path_prefix}_*")

    def load_model(self, path_prefix, llm_encoder=None, gnn_encoder=None,
                   load_optimizer=False):
        """Strictly load every state-producing component."""
        import os
        required = ['fusion', 'actor', 'critic', 'llm', 'gnn']
        missing = [name for name in required if not os.path.exists(f"{path_prefix}_{name}")]
        if missing:
            raise FileNotFoundError(f"incomplete checkpoint, missing: {', '.join(missing)}")
        if llm_encoder is None or gnn_encoder is None:
            raise ValueError('llm_encoder and gnn_encoder are required for loading')
        self.fusion.load_state_dict(
            torch.load(f"{path_prefix}_fusion", map_location=self.device))
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor", map_location=self.device))
        self.critic.load_state_dict(
            torch.load(f"{path_prefix}_critic", map_location=self.device))
        llm_encoder.load_state_dict(
            torch.load(f"{path_prefix}_llm", map_location=self.device))
        gnn_encoder.load_state_dict(
            torch.load(f"{path_prefix}_gnn", map_location=self.device))
        training_path = f"{path_prefix}_training"
        if load_optimizer and os.path.exists(training_path):
            state = torch.load(training_path, map_location=self.device)
            self.optimizer.load_state_dict(state['optimizer'])
            self.train_step = state.get('train_step', 0)
        print(f"[PPO] 模型已从 {path_prefix}_* 加载")


def create_pruning_agent(embed_dim=128, device='cpu'):
    """工厂函数: 创建 PruningPPO 实例"""
    return PruningPPO(embed_dim=embed_dim, device=device)
