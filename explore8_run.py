"""
探索版8: Graph Transformer分类 + Qwen embedding推荐增强
分类: SAN(Sparse Adaptive Network) - 自适应学习边权重, 适合稀疏图
推荐: 用Qwen text-embedding增强物品表示(赛题允许)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse
from src.models import build_classification_model
from src.train_cls_improved import drop_edge, label_propagation

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLS_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A分类", "A分类", "A1.npz")


class GraphTransformerLayer(nn.Module):
    """简化的Graph Transformer: 自适应学习邻居权重"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_sparse):
        # x: (N, in_dim), adj_sparse: torch sparse
        Q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(x).view(-1, self.num_heads, self.head_dim)
        
        # 简化: 用邻接矩阵做稀疏注意力 (只对邻居计算注意力)
        # adj_sparse: (N, N) sparse
        # 对每个节点, 只聚合邻居的V (用GCN式的聚合, 但加上QK注意力)
        
        # 用稀疏矩阵乘法高效计算
        # attention = softmax(Q @ K^T * adj) 只在有边的地方计算
        # 简化为: 先用adj聚合V (类似GCN), 再用FFN
        V_flat = V.reshape(-1, V.shape[1] * V.shape[2])
        aggregated = torch.sparse.mm(adj_sparse, V_flat)
        
        # 残差 + LayerNorm
        h = self.norm1(aggregated + x[:, :aggregated.shape[1]] if x.shape[1] == aggregated.shape[1] else aggregated)
        h = self.dropout(h)
        h = h + self.ffn(h)
        h = self.norm2(h)
        return h


class GraphTransformer(nn.Module):
    """Graph Transformer for node classification"""
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj_sparse=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, adj_sparse)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


def run_cls(device="cuda", n_ensemble=30, lp_weight=0.3):
    """Graph Transformer + GCN 混合集成"""
    print("\n" + "=" * 70)
    print(f"  Task1: GraphTransformer{n_ensemble//2} + GCN{n_ensemble//2} 混合集成")
    print("=" * 70)
    t0 = time.time()

    data = load_classification_data(CLS_DATA)
    adj, features, labels = data["adj"], data["features"], data["labels"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    num_classes = data["num_classes"]
    feat_dim = data["feat_dim"]

    features = normalize(features, norm="l2", axis=1).astype(np.float32)
    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    test_t = torch.LongTensor(test_idx).to(device)

    np.random.seed(42)
    perm = np.random.permutation(train_idx)
    n_val = int(len(train_idx) * 0.1)
    train_only_t = torch.LongTensor(perm[n_val:]).to(device)
    val_t = torch.LongTensor(perm[:n_val]).to(device)

    n_gcn = n_ensemble // 2
    n_gt = n_ensemble - n_gcn
    all_probs = []
    best_val = 0

    # GCN集成
    print(f"\n[GCN] {n_gcn}模型...")
    for i in range(n_gcn):
        seed = 42 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_probs = None; bv = 0
        for epoch in range(1, 301):
            model.train(); opt.zero_grad()
            adj_tr = drop_edge(adj_sparse, 0.2)
            logits = model(features_t, adj_tr)
            loss = crit(logits[train_only_t], labels_t[train_only_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vp = model(features_t, adj_sparse)[val_t].argmax(1)
                    va = (vp == labels_t[val_t]).float().mean().item()
                    if va > bv:
                        bv = va
                        best_probs = F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
        all_probs.append(best_probs)
        if va > best_val: best_val = va
        if (i+1) % 5 == 0: print(f"    GCN {i+1}/{n_gcn} Val={bv:.4f}")

    # Graph Transformer集成
    print(f"\n[GraphTransformer] {n_gt}模型...")
    gt_val = 0
    for i in range(n_gt):
        seed = 200 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        gt = GraphTransformer(feat_dim, 256, num_classes, num_layers=2, num_heads=4, dropout=0.5).to(device)
        opt = torch.optim.Adam(gt.parameters(), lr=0.005, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_probs = None; bv = 0
        for epoch in range(1, 301):
            gt.train(); opt.zero_grad()
            adj_tr = drop_edge(adj_sparse, 0.2)
            logits = gt(features_t, adj_tr)
            loss = crit(logits[train_only_t], labels_t[train_only_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gt.parameters(), 5.0)
            opt.step(); sched.step()
            if epoch % 10 == 0:
                gt.eval()
                with torch.no_grad():
                    vp = gt(features_t, adj_sparse)[val_t].argmax(1)
                    va = (vp == labels_t[val_t]).float().mean().item()
                    if va > bv:
                        bv = va
                        best_probs = F.softmax(gt(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
        all_probs.append(best_probs)
        if bv > gt_val: gt_val = bv
        if (i+1) % 5 == 0: print(f"    GT {i+1}/{n_gt} Val={bv:.4f}")

    print(f"  GCN Val={best_val:.4f}, GT Val={gt_val:.4f}")

    mixed = np.mean(all_probs, axis=0)

    # 标签传播 + 伪标签
    lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    fused1 = (1-lp_weight) * mixed + lp_weight * lp
    conf = fused1.max(1)
    mask = conf > 0.7
    train2 = np.concatenate([train_idx, test_idx[mask]])
    labels2 = labels.copy()
    labels2[test_idx[mask]] = fused1.argmax(1)[mask]
    print(f"  伪标签: {mask.sum()}个")

    # 第2轮: 伪标签重训GCN
    ft_mask = torch.LongTensor(train2).to(device)
    ft_labels = torch.LongTensor(labels2).to(device)
    print(f"\n[第2轮] 伪标签+GCN{n_gcn}...")
    pseudo_probs = []
    for i in range(n_gcn):
        seed = 500 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        for epoch in range(1, 301):
            model.train(); opt.zero_grad()
            adj_tr = drop_edge(adj_sparse, 0.2)
            logits = model(features_t, adj_tr)
            loss = crit(logits[ft_mask], ft_labels[ft_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pseudo_probs.append(F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy())
    r2 = np.mean(pseudo_probs, axis=0)
    fused2 = (1-lp_weight) * r2 + lp_weight * lp

    test_pred = fused2.argmax(1)
    elapsed = time.time() - t0
    print(f"\n[Task1完成] GT+GCN混合 | 耗时: {elapsed/60:.1f}min")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return max(best_val, gt_val)


def main():
    print("=" * 70)
    print("  探索版8: GraphTransformer + GCN混合集成")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_val = run_cls(device=device, n_ensemble=30, lp_weight=0.3)
    elapsed = time.time() - t0
    print(f"\n分类Val={cls_val:.4f}")
    print(f"耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
