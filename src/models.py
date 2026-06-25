"""
模型定义模块
- GraphSAGE: 图节点分类模型
- GRU4Rec: 序列推荐模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== Task1: GraphSAGE 节点分类模型 ========================

class GraphSAGE(nn.Module):
    """GraphSAGE 模型 (mean aggregation)
    使用稀疏邻接矩阵进行消息传递，避免 OOM
    """

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2,
                 dropout=0.5, normalization="sym"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalization = normalization

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim * 2, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj_sparse):
        """前向传播
        x: (N, in_dim) 节点特征
        adj_sparse: torch sparse COO 归一化邻接矩阵
        """
        h = x
        for i in range(self.num_layers):
            # 聚合邻居信息: adj @ h
            h_neigh = torch.sparse.mm(adj_sparse, h)

            if self.normalization == "rw":
                # 随机游走归一化已在邻接矩阵中处理
                pass

            # 拼接自身特征和邻居特征
            h_concat = torch.cat([h, h_neigh], dim=-1)
            h = self.layers[i](h_concat)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)
        return logits


class GCN(nn.Module):
    """改进版 GCN 模型 (谱域卷积) + BatchNorm + 残差连接
    A_norm = D^(-1/2) (A+I) D^(-1/2)
    H' = A_norm @ H @ W
    """

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2,
                 dropout=0.5, use_residual=True, use_batchnorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity())

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity())

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj_sparse):
        h = x
        for i in range(self.num_layers):
            h_in = h
            h_neigh = torch.sparse.mm(adj_sparse, h)
            h = self.layers[i](h_neigh)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # 残差连接 (维度匹配时)
            if self.use_residual and h_in.shape[-1] == h.shape[-1]:
                h = h + h_in

        logits = self.classifier(h)
        return logits


class GAT(nn.Module):
    """简化版 GAT (基于注意力机制的图卷积)
    使用稀疏邻接矩阵
    """

    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=4,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()

        # 第一层
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.attn_layers.append(nn.Linear(hidden_dim * 2, 1))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.attn_layers.append(nn.Linear(hidden_dim * 2, 1))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj_sparse):
        h = x
        for i in range(self.num_layers):
            h_proj = self.layers[i](h)

            # 计算注意力权重 (简化版: 基于特征相似度)
            # 使用邻接矩阵限制注意力只计算邻居之间
            h_neigh = torch.sparse.mm(adj_sparse, h_proj)

            # 混合自身和邻居 (注意力简化为均值)
            h = F.relu(h_proj + h_neigh)
            h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)
        return logits


def build_classification_model(model_type, in_dim, hidden_dim, num_classes,
                               num_layers, dropout, normalization="sym"):
    """根据模型类型构建分类模型"""
    if model_type == "sage":
        return GraphSAGE(in_dim, hidden_dim, num_classes, num_layers, dropout, normalization)
    elif model_type == "gcn":
        return GCN(in_dim, hidden_dim, num_classes, num_layers, dropout)
    elif model_type == "gat":
        return GAT(in_dim, hidden_dim, num_classes, num_heads=4, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ======================== Task2: GRU4Rec 序列推荐模型 ========================

class GRU4Rec(nn.Module):
    """GRU4Rec 序列推荐模型
    输入: item 序列
    输出: 每个位置对全部 item 的打分
    """

    def __init__(self, num_items, embedding_dim=64, hidden_dim=128,
                 num_layers=1, dropout=0.2, max_len=50):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Item embedding (0 为 padding)
        self.item_embedding = nn.Embedding(
            num_items + 2, embedding_dim, padding_idx=0
        )

        # GRU
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_tensor, seq_lengths):
        """前向传播
        seq_tensor: (batch, max_len) item id 序列
        seq_lengths: (batch,) 每个序列的实际长度
        返回: (batch, num_items+1) 最后一个位置的打分
        """
        # Embedding
        emb = self.item_embedding(seq_tensor)  # (batch, max_len, embed_dim)
        emb = self.dropout(emb)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.gru(packed)
        # hidden: (num_layers, batch, hidden_dim)
        out = hidden[-1]  # 取最后一层: (batch, hidden_dim)
        out = self.dropout(out)

        # 输出打分
        scores = self.output_proj(out)  # (batch, num_items+1)
        return scores


class SASRec(nn.Module):
    """简化版 SASRec (Self-Attentive Sequential Recommendation)
    基于 Transformer 的序列推荐
    """

    def __init__(self, num_items, embedding_dim=64, hidden_dim=128,
                 num_heads=2, num_layers=2, dropout=0.2, max_len=50):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len

        self.item_embedding = nn.Embedding(
            num_items + 2, embedding_dim, padding_idx=0
        )
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_proj = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, seq_tensor, seq_lengths):
        batch_size, seq_len = seq_tensor.shape

        # Item + Position embedding
        emb = self.item_embedding(seq_tensor)
        pos = torch.arange(seq_len, device=seq_tensor.device).unsqueeze(0).expand(batch_size, -1)
        emb = emb + self.pos_embedding(pos)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        # Transformer (需要 mask padding)
        padding_mask = (seq_tensor == 0)
        out = self.transformer(emb, src_key_padding_mask=padding_mask)

        # 取最后一个非 padding 位置
        idx = (seq_lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, out.size(-1))
        out = out.gather(1, idx).squeeze(1)

        scores = self.output_proj(out)
        return scores


def build_recommendation_model(model_type, num_items, embedding_dim=64,
                                hidden_dim=128, num_layers=1, dropout=0.2,
                                max_len=50):
    """根据模型类型构建推荐模型"""
    if model_type == "gru4rec":
        return GRU4Rec(num_items, embedding_dim, hidden_dim, num_layers, dropout, max_len)
    elif model_type == "sasrec":
        return SASRec(num_items, embedding_dim, hidden_dim, num_heads=2,
                      num_layers=2, dropout=dropout, max_len=max_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
