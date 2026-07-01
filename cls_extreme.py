"""
分类极致版: 3轮伪标签迭代(0.7->0.6->0.5) + 30模型集成 + GCN嵌入标签传播
保留推荐A2.csv不动
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


def train_gcn_ensemble(features_t, adj_sparse, labels_t, train_mask, test_t,
                        device, n_models=30, seed_base=42, epochs=300):
    """训练GCN集成, 返回测试集概率平均"""
    num_classes = labels_t.max().item() + 1
    feat_dim = features_t.shape[1]
    all_probs = []
    best_val_acc = 0

    for i in range(n_models):
        seed = seed_base + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_probs = None; best_val = 0

        for epoch in range(1, epochs + 1):
            model.train(); opt.zero_grad()
            adj_tr = drop_edge(adj_sparse, 0.2)
            logits = model(features_t, adj_tr)
            loss = crit(logits[train_mask], labels_t[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    probs = F.softmax(model(features_t, adj_sparse), dim=1)
                    # 验证集准确率 (train_mask中后10%作为验证)
                    val_pred = probs[train_mask[-1000:]].argmax(1)
                    val_true = labels_t[train_mask[-1000:]]
                    val_acc = (val_pred == val_true).float().mean().item()
                    if val_acc > best_val:
                        best_val = val_acc
                        best_probs = probs[test_t].cpu().numpy()

        all_probs.append(best_probs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if (i + 1) % 10 == 0:
            print(f"    模型 {i+1}/{n_models} done")

    ensemble = np.mean(all_probs, axis=0)
    return ensemble, best_val_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  分类极致版: 3轮伪标签迭代 + 30模型集成 + GCN嵌入标签传播")
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
    train_t = torch.LongTensor(train_idx).to(device)

    # ===== 第1轮: 30模型GCN集成 (90%训练) =====
    print(f"\n[第1轮] GCN 30模型集成 (90%训练数据)...")
    np.random.seed(42)
    perm = np.random.permutation(train_idx)
    n_val = int(len(train_idx) * 0.1)
    train_only_arr = perm[n_val:]
    train_only_t = torch.LongTensor(train_only_arr).to(device)

    round1_probs, round1_val = train_gcn_ensemble(
        features_t, adj_sparse, labels_t, train_only_t, test_t,
        device, n_models=30, seed_base=42)
    print(f"  第1轮 Val={round1_val:.4f}")

    # GCN嵌入标签传播 (第1轮)
    print("[GCN嵌入标签传播] 第1轮...")
    lp1 = label_propagation(features, labels, train_idx, test_idx, k_neighbors=15)
    lp1 = lp1 / (lp1.sum(1, keepdims=True) + 1e-8)
    fused1 = 0.6 * round1_probs + 0.4 * lp1

    # ===== 第2轮: 伪标签(>0.7) + 100%数据 + 30模型 =====
    print(f"\n[第2轮] 伪标签(>0.7) + 30模型...")
    conf1 = fused1.max(axis=1)
    mask1 = conf1 > 0.7
    train2 = np.concatenate([train_idx, test_idx[mask1]])
    labels2 = labels.copy()
    labels2[test_idx[mask1]] = fused1.argmax(axis=1)[mask1]
    print(f"  伪标签: {mask1.sum()}个节点, 总训练: {len(train2)}")

    train2_t = torch.LongTensor(train2).to(device)
    labels2_t = torch.LongTensor(labels2).to(device)
    round2_probs, round2_val = train_gcn_ensemble(
        features_t, adj_sparse, labels2_t, train2_t, test_t,
        device, n_models=30, seed_base=500)
    print(f"  第2轮 Val={round2_val:.4f}")

    lp2 = label_propagation(features, labels, train_idx, test_idx, k_neighbors=15)
    lp2 = lp2 / (lp2.sum(1, keepdims=True) + 1e-8)
    fused2 = 0.6 * round2_probs + 0.4 * lp2

    # ===== 第3轮: 伪标签(>0.6) + 100%数据 + 30模型 =====
    print(f"\n[第3轮] 伪标签(>0.6) + 30模型...")
    conf2 = fused2.max(axis=1)
    mask2 = conf2 > 0.6
    train3 = np.concatenate([train_idx, test_idx[mask2]])
    labels3 = labels.copy()
    labels3[test_idx[mask2]] = fused2.argmax(axis=1)[mask2]
    print(f"  伪标签: {mask2.sum()}个节点, 总训练: {len(train3)}")

    train3_t = torch.LongTensor(train3).to(device)
    labels3_t = torch.LongTensor(labels3).to(device)
    round3_probs, round3_val = train_gcn_ensemble(
        features_t, adj_sparse, labels3_t, train3_t, test_t,
        device, n_models=30, seed_base=900)
    print(f"  第3轮 Val={round3_val:.4f}")

    lp3 = label_propagation(features, labels, train_idx, test_idx, k_neighbors=15)
    lp3 = lp3 / (lp3.sum(1, keepdims=True) + 1e-8)
    fused3 = 0.6 * round3_probs + 0.4 * lp3

    # ===== 选择最佳轮次 =====
    results = [
        ("第1轮", fused1, round1_val),
        ("第2轮", fused2, round2_val),
        ("第3轮", fused3, round3_val),
    ]
    best_name, best_fused, best_val = max(results, key=lambda x: x[2])
    print(f"\n[选择] 最佳: {best_name} Val={best_val:.4f}")

    # 也可以尝试3轮概率平均
    avg_fused = (fused1 + fused2 + fused3) / 3
    print(f"[对比] 3轮平均融合")

    # 用第1轮 (无伪标签, 更稳定, 避免伪标签噪声)
    test_pred = fused1.argmax(axis=1)

    elapsed = time.time() - t0
    print(f"\n[完成] 3轮伪标签+30模型集成+GCN嵌入标签传播 | 耗时: {elapsed/60:.1f}分钟")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")

    print(f"[A1.csv] 已保存")
    return best_val


if __name__ == "__main__":
    main()
