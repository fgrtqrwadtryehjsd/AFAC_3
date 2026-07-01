"""
探索版7: 分类用特征KNN图替代原始图 + 推荐用物品流行度后处理
核心创新: 53%孤立节点没有图结构信息, 用特征相似度构建KNN图为它们创造邻居
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse
from src.models import build_classification_model
from src.train_cls_improved import drop_edge, label_propagation

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLS_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A分类", "A分类", "A1.npz")


def build_feature_knn_graph(features, original_adj, k=5):
    """用特征相似度构建KNN图, 与原图融合
    为孤立节点(度<=1)创造邻居连接
    """
    print(f"[KNN图] 构建k={k}近邻图...")
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=-1).fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    n = features.shape[0]
    rows, cols = [], []
    for i in range(n):
        for j in indices[i][1:]:  # 跳过自身
            rows.append(i); cols.append(j)
    
    knn_adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    knn_adj = knn_adj.maximum(knn_adj.T)
    
    # 融合: 原图 ∪ KNN图
    fused = original_adj.maximum(knn_adj)
    degree = np.array(fused.sum(axis=1)).flatten()
    isolated = (degree <= 1).sum()
    print(f"[KNN图] 原图边数={original_adj.nnz}, KNN边数={knn_adj.nnz}, 融合={fused.nnz}")
    print(f"[KNN图] 融合后孤立节点: {isolated}/{n} ({isolated/n:.1%})")
    return fused


def run_cls(device="cuda", n_ensemble=50, lp_weight=0.3, use_knn=True, knn_k=5):
    print("\n" + "=" * 70)
    print(f"  Task1: GCN{n_ensemble} + KNN图(k={knn_k}) + 伪标签 + lp={lp_weight}")
    print("=" * 70)
    t0 = time.time()

    data = load_classification_data(CLS_DATA)
    adj, features, labels = data["adj"], data["features"], data["labels"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    num_classes = data["num_classes"]
    feat_dim = data["feat_dim"]

    features = normalize(features, norm="l2", axis=1).astype(np.float32)
    
    # KNN图增强
    if use_knn:
        adj = build_feature_knn_graph(features, adj, k=knn_k)
    
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

    print(f"\n[第1轮] GCN {n_ensemble}模型...")
    all_probs = []
    best_val = 0
    for i in range(n_ensemble):
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
        if (i+1) % 10 == 0: print(f"    {i+1}/{n_ensemble} Val={bv:.4f}")
    r1 = np.mean(all_probs, axis=0)
    print(f"  R1 Val={best_val:.4f}")

    lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    fused1 = (1-lp_weight) * r1 + lp_weight * lp
    conf = fused1.max(1)
    mask = conf > 0.7
    train2 = np.concatenate([train_idx, test_idx[mask]])
    labels2 = labels.copy()
    labels2[test_idx[mask]] = fused1.argmax(1)[mask]
    print(f"  伪标签: {mask.sum()}个")

    ft_mask = torch.LongTensor(train2).to(device)
    ft_labels = torch.LongTensor(labels2).to(device)
    print(f"\n[第2轮] 伪标签+{n_ensemble}模型...")
    pseudo_probs = []
    for i in range(n_ensemble):
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
    print(f"\n[Task1完成] KNN图增强 | 耗时: {elapsed/60:.1f}min")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return best_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  探索版7: KNN图增强(k={args.knn_k})分类")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_val = run_cls(device=device, n_ensemble=50, lp_weight=0.3, use_knn=True, knn_k=args.knn_k)
    elapsed = time.time() - t0
    print(f"\n分类Val={cls_val:.4f}")
    print(f"耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
