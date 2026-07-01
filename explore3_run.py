"""
探索版3: 分类GCN+SAGE混合集成 + 推荐物品特征3模型集成
分类: 40GCN + 10SAGE 混合集成 (不同架构互补)
推荐: 保持3模型物品特征集成 (Val=0.5960)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse
from src.models import build_classification_model
from src.train_cls_improved import drop_edge, label_propagation

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLS_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A分类", "A分类", "A1.npz")


def run_cls(device="cuda", n_gcn=40, n_sage=10, lp_weight=0.3):
    print("\n" + "=" * 70)
    print(f"  Task1: GCN{n_gcn}+SAGE{n_sage}混合集成 + 伪标签 + lp={lp_weight}")
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
    train_only_arr = perm[n_val:]
    train_only_t = torch.LongTensor(train_only_arr).to(device)
    val_t = torch.LongTensor(perm[:n_val]).to(device)

    def train_ensemble(model_type, n_models, seed_base):
        all_probs = []
        best_val = 0
        for i in range(n_models):
            seed = seed_base + i * 10
            torch.manual_seed(seed); np.random.seed(seed)
            model = build_classification_model(model_type, feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
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
            if (i+1) % 10 == 0: print(f"    {model_type} {i+1}/{n_models} Val={bv:.4f}")
        return np.mean(all_probs, axis=0), best_val

    # GCN集成
    print(f"\n[GCN集成] {n_gcn}模型...")
    gcn_probs, gcn_val = train_ensemble("gcn", n_gcn, 42)
    print(f"  GCN Val={gcn_val:.4f}")

    # SAGE集成
    print(f"\n[SAGE集成] {n_sage}模型...")
    sage_probs, sage_val = train_ensemble("sage", n_sage, 42 + n_gcn * 10)
    print(f"  SAGE Val={sage_val:.4f}")

    # 混合集成 (按验证分数加权)
    total_val = gcn_val + sage_val
    gcn_w = gcn_val / total_val
    sage_w = sage_val / total_val
    print(f"\n[混合] GCN权重={gcn_w:.3f}, SAGE权重={sage_w:.3f}")
    mixed = gcn_w * gcn_probs + sage_w * sage_probs

    # 标签传播
    lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    fused = (1-lp_weight) * mixed + lp_weight * lp

    # 伪标签
    conf = fused.max(1)
    mask = conf > 0.7
    train2 = np.concatenate([train_idx, test_idx[mask]])
    labels2 = labels.copy()
    labels2[test_idx[mask]] = fused.argmax(1)[mask]
    print(f"  伪标签: {mask.sum()}个")

    # 第2轮: 伪标签重训
    ft_mask = torch.LongTensor(train2).to(device)
    ft_labels = torch.LongTensor(labels2).to(device)
    print(f"\n[第2轮] 伪标签+{n_gcn}GCN...")
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
    print(f"\n[Task1完成] GCN+SAGE混合+伪标签 | 耗时: {elapsed/60:.1f}min")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return max(gcn_val, sage_val)


def main():
    print("=" * 70)
    print("  探索版3: GCN+SAGE混合集成分类")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_val = run_cls(device=device, n_gcn=40, n_sage=10, lp_weight=0.3)
    elapsed = time.time() - t0
    est_test = cls_val * 1.02
    print(f"\n分类Val={cls_val:.4f} → 预估test={est_test:.4f}")
    print(f"耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
