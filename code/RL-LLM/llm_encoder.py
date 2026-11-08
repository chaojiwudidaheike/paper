"""
LLM 文本特征编码器

将当前调度状态转化为自然语言文本描述，
通过预训练嵌入模型或 PyTorch Transformer 转化为特征向量。

输出: z_LLM, dim=128
"""

import numpy as np
import torch
import torch.nn as nn


def build_text_prompt(Agent_Data, Env, centerRow, centerLin, Center_Scope,
                      CurrTimeAge, SingleStep, AllTime, Task_Time):
    """Build a fixed-size, scale-free state summary for the text encoder."""
    ROW = Env.shape[1]
    LIN = Env.shape[2]
    now = CurrTimeAge * SingleStep
    online = ((Agent_Data[:, 1] <= now) & (now < Agent_Data[:, 4]))
    in_scope = ((Agent_Data[:, 2] - 1 >= centerRow) &
                (Agent_Data[:, 2] - 1 < centerRow + Center_Scope) &
                (Agent_Data[:, 3] - 1 >= centerLin) &
                (Agent_Data[:, 3] - 1 < centerLin + Center_Scope))
    active = online & in_scope
    counts = [int(np.sum(active & (Agent_Data[:, 5] == kind))) for kind in (1, 2, 3)]
    uavs = Agent_Data[active & (Agent_Data[:, 5] == 3)]
    battery = float(np.mean(uavs[:, 7] / np.maximum(uavs[:, 8], 1e-8))) if len(uavs) else 0.0
    task_mask = Env[1] > 0
    task_cost = float(np.mean(Task_Time[task_mask]) / max(AllTime, 1)) if np.any(task_mask) else 0.0
    area = max(ROW * LIN, 1)
    values = {
        "progress": min(now / max(AllTime, 1), 1.0),
        "scope_h": Center_Scope / max(ROW, 1),
        "scope_w": Center_Scope / max(LIN, 1),
        "workers": counts[0] / max(int(np.sum(active)), 1),
        "cars": counts[1] / max(int(np.sum(active)), 1),
        "uavs": counts[2] / max(int(np.sum(active)), 1),
        "tasks": float(np.sum(task_mask)) / area,
        "charges": float(np.sum(Env[0] > 0)) / area,
        "battery": battery,
        "task_cost": task_cost,
    }
    return "调度状态 " + " ".join(f"{key}={value:.4f}" for key, value in values.items())


class SimpleTextEncoder(nn.Module):
    """
    基于 PyTorch TransformerEncoder 的简单文本编码器。
    使用字符级编码以避免依赖外部 tokenizer。

    输入: 文本字符串 (通过 build_text_prompt 生成)
    输出: 128-dim 特征向量
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, max_len=2048):
        super().__init__()
        # 使用 Unicode 编码范围覆盖中英文字符
        self.vocab_size = 65536  # 直接用 char code 作为索引
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.char_embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, text_list):
        """
        Args:
            text_list: list of str, 批量的文本描述
        Returns:
            z_LLM: (batch_size, embed_dim) 特征向量
        """
        # 将字符串转为字符索引
        batch_size = len(text_list)
        device = self.char_embedding.weight.device

        # 编码每个字符为 Unicode code point
        sequence_len = max(1, min(max(map(len, text_list), default=1), self.max_len))
        char_ids = torch.zeros(batch_size, sequence_len, dtype=torch.long, device=device)
        for b, text in enumerate(text_list):
            # 截断到 max_len
            text = text[:sequence_len]
            for c, ch in enumerate(text):
                char_ids[b, c] = min(ord(ch), self.vocab_size - 1)

        # 字符嵌入 + 位置嵌入
        x = self.char_embedding(char_ids)  # (B, L, E)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 创建 padding mask (char_id == 0 的位置)
        src_key_padding_mask = (char_ids == 0)

        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 对非 padding 位置做 mean pooling
        valid_mask = (char_ids != 0).float().unsqueeze(-1)  # (B, L, 1)
        x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B, E)

        # 输出投影
        z_LLM = self.output_proj(x)  # (B, embed_dim)

        return z_LLM


# ============================================================
# 工厂函数：尝试加载 sentence-transformers，失败则用 SimpleTextEncoder
# ============================================================

_HAS_SENTENCE_TRANSFORMERS = None


def _check_sentence_transformers():
    global _HAS_SENTENCE_TRANSFORMERS
    if _HAS_SENTENCE_TRANSFORMERS is None:
        try:
            import sentence_transformers  # noqa: F401
            _HAS_SENTENCE_TRANSFORMERS = True
        except ImportError:
            _HAS_SENTENCE_TRANSFORMERS = False
    return _HAS_SENTENCE_TRANSFORMERS


class SentenceTransformerEncoder(nn.Module):
    """
    使用 sentence-transformers 预训练模型的编码器。
    模型输出维度会通过投影层映射到 embed_dim=128。
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', embed_dim=128):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.st_model = SentenceTransformer(model_name)
        # 冻结预训练模型
        for param in self.st_model.parameters():
            param.requires_grad = False
        st_dim = self.st_model.get_sentence_embedding_dimension()
        self.proj = nn.Linear(st_dim, embed_dim)

    def forward(self, text_list):
        device = self.proj.weight.device
        # sentence-transformers 默认在 CPU 上运行
        embeddings = self.st_model.encode(
            text_list,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if embeddings.device != device:
            embeddings = embeddings.to(device)
        z_LLM = self.proj(embeddings)
        return z_LLM


def create_llm_encoder(embed_dim=128, use_pretrained=True):
    """
    创建 LLM 编码器。

    Args:
        embed_dim: 输出特征维度
        use_pretrained: 是否尝试使用 sentence-transformers 预训练模型

    Returns:
        nn.Module 编码器
    """
    if use_pretrained and _check_sentence_transformers():
        print("[LLM Encoder] 使用 sentence-transformers 预训练模型")
        return SentenceTransformerEncoder(embed_dim=embed_dim)
    else:
        if use_pretrained:
            print("[LLM Encoder] sentence-transformers 不可用，回退到 SimpleTextEncoder")
        else:
            print("[LLM Encoder] 使用 SimpleTextEncoder (纯 PyTorch)")
        return SimpleTextEncoder(embed_dim=embed_dim)
