"""
冲刺0.65版运行脚本
分类: 3轮伪标签 + GCN/SAGE混合20集成 + 标签传播(0.4)
推荐: GRU4Rec + 物品协同过滤(CF)混合, 冷启动用户用CF
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize
import scipy.sparse as sp
from collections import Counter
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_classification_model, build_recommendation_model
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.train_cls_improved import drop_edge, label_propagation
from src.train_rec import RecDataset, TestRecDataset
from src.train_rec_improved import compute_ndcg_at_k

CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ======================== 分类: 3轮伪标签 + 混合集成 ========================

def train_cls_optimized(npz_path, device="cuda", n_ensemble=20):
    """分类: 3轮伪标签 + GCN/SAGE混合集成"""

    print("\n" + "=" * 70)
    print("  Task1: 3轮伪标签 + GCN/SAGE混合20集成 + 标签传播(0.4)")
    print("=" * 70)
    start_time = time.time()

    data = load_classification_data(npz_path)
    adj = data["adj"]
    features = sk_normalize(data["features"], norm="l2", axis=1).astype(np.float32)
    labels = data["labels"].copy()
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    num_nodes, feat_dim = features.shape
    num_classes = int(labels.max()) + 1

    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")

    # 划分验证集
    train_labels_arr = labels[train_idx]
    train_sub, val_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=42,
        stratify=train_labels_arr if len(np.unique(train_labels_arr)) > 1 else None,
    )
    val_idx = train_idx[val_sub]
    train_only_idx = train_idx[train_sub]

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)
    val_mask_t = torch.LongTensor(val_idx).to(device)
    test_idx_t = torch.LongTensor(test_idx).to(device)

    def train_ensemble(train_indices, train_labels_arr, n_models, model_types=None):
        """训练一组集成模型"""
        if model_types is None:
            # 混合架构: 60% GCN + 40% SAGE
            model_types = ["gcn"] * int(n_models * 0.6) + ["sage"] * (n_models - int(n_models * 0.6))

        train_mask_t = torch.LongTensor(train_indices).to(device)
        train_labels_t = torch.LongTensor(train_labels_arr).to(device)
        ensemble_logits = []
        val_accs = []

        for i in range(n_models):
            seed = 42 + i * 10
            torch.manual_seed(seed)
            np.random.seed(seed)
            mtype = model_types[i % len(model_types)]

            model = build_classification_model(
                mtype, feat_dim, 256, num_classes, 2, 0.5, "sym"
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            best_test_logits = None

            for epoch in range(1, 301):
                model.train()
                optimizer.zero_grad()
                adj_train = drop_edge(adj_sparse, 0.2)
                logits = model(features_t, adj_train)
                loss = criterion(logits[train_mask_t], train_labels_t[train_mask_t])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    logits = model(features_t, adj_sparse)
                    val_pred = logits[val_mask_t].argmax(dim=1).cpu().numpy()
                    val_true = labels[val_idx]
                    val_acc = accuracy_score(val_true, val_pred)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    with torch.no_grad():
                        best_test_logits = F.softmax(
                            model(features_t, adj_sparse)[test_idx_t], dim=1
                        ).cpu().numpy()

            ensemble_logits.append(best_test_logits)
            val_accs.append(best_val_acc)

        return np.mean(ensemble_logits, axis=0), np.mean(val_accs), val_accs

    # ===== 轮次1: 原始训练集 =====
    print(f"\n[轮次1] 原始训练集 ({n_ensemble} 模型, GCN+SAGE混合)")
    pred_1, val_1, accs_1 = train_ensemble(train_only_idx, labels, n_ensemble)
    print(f"  Val Acc = {val_1:.4f}, 各模型: {[f'{a:.4f}' for a in accs_1[:5]]}...")

    # ===== 轮次2: 伪标签 (置信度>0.8) =====
    test_pred_1 = pred_1.argmax(axis=1)
    conf_1 = pred_1.max(axis=1)
    mask_1 = conf_1 > 0.8
    print(f"\n[轮次2] 伪标签 (置信度>0.8): {mask_1.sum()}/{len(test_idx)} ({mask_1.mean()*100:.1f}%)")

    exp_train_2 = np.concatenate([train_only_idx, test_idx[mask_1]])
    exp_labels_2 = labels.copy()
    exp_labels_2[test_idx[mask_1]] = test_pred_1[mask_1]

    pred_2, val_2, accs_2 = train_ensemble(exp_train_2, exp_labels_2, n_ensemble)
    print(f"  Val Acc = {val_2:.4f} (轮次1: {val_1:.4f}, 变化: {val_2-val_1:+.4f})")

    # ===== 轮次3: 伪标签 (置信度>0.7) =====
    test_pred_2 = pred_2.argmax(axis=1)
    conf_2 = pred_2.max(axis=1)
    mask_2 = conf_2 > 0.7
    print(f"\n[轮次3] 伪标签 (置信度>0.7): {mask_2.sum()}/{len(test_idx)} ({mask_2.mean()*100:.1f}%)")

    exp_train_3 = np.concatenate([train_only_idx, test_idx[mask_2]])
    exp_labels_3 = labels.copy()
    exp_labels_3[test_idx[mask_2]] = test_pred_2[mask_2]

    pred_3, val_3, accs_3 = train_ensemble(exp_train_3, exp_labels_3, n_ensemble)
    print(f"  Val Acc = {val_3:.4f} (轮次2: {val_2:.4f}, 变化: {val_3-val_2:+.4f})")

    # 选择最佳轮次
    results = [(pred_1, val_1, "轮次1"), (pred_2, val_2, "轮次2"), (pred_3, val_3, "轮次3")]
    best_pred, best_val, best_name = max(results, key=lambda x: x[1])
    print(f"\n[选择] {best_name} (Val Acc = {best_val:.4f})")

    # ===== 标签传播后处理 =====
    print(f"\n[后处理] 标签传播 (权重=0.4)")
    lp_logits = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp_logits = lp_logits / (lp_logits.sum(axis=1, keepdims=True) + 1e-8)
    final_logits = 0.6 * best_pred + 0.4 * lp_logits
    test_pred = final_logits.argmax(axis=1)

    elapsed = time.time() - start_time
    print(f"\n[Task1] 完成 | Val Acc = {best_val:.4f} | 耗时: {elapsed:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, pred in zip(test_idx, test_pred):
            f.write(f"{idx},{pred}\n")

    return best_val


# ======================== 推荐: GRU4Rec + 物品协同过滤 ========================

def train_rec_with_cf(data_dir, device="cuda"):
    """推荐: GRU4Rec + 物品协同过滤(CF)混合"""

    print("\n" + "=" * 70)
    print("  Task2: GRU4Rec + 物品协同过滤(CF)混合")
    print("=" * 70)
    start_time = time.time()

    data = load_recommendation_data(data_dir)
    train_df = data["train_df"]
    test_df = data["test_df"]
    item2idx = data["item2idx"]
    idx2item = data["idx2item"]
    num_items = data["num_items"]

    max_seq_len = 50
    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_seq_len
    )

    # 划分训练/验证集
    n_val = int(len(train_seqs) * 0.1)
    np.random.seed(42)
    indices = np.random.permutation(len(train_seqs))
    val_seqs = [train_seqs[i] for i in indices[:n_val]]
    val_targets = [train_targets[i] for i in indices[:n_val]]
    tr_seqs = [train_seqs[i] for i in indices[n_val:]]
    tr_targets = [train_targets[i] for i in indices[n_val:]]

    # 序列增强
    aug_seqs, aug_targets = list(tr_seqs), list(tr_targets)
    for seq, target in zip(tr_seqs, tr_targets):
        if len(seq) > 5:
            for trunc_len in [5, 10]:
                if len(seq) > trunc_len:
                    aug_seqs.append(seq[-trunc_len:])
                    aug_targets.append(target)
    print(f"[增强] {len(tr_seqs)} → {len(aug_seqs)}")

    # ===== 1. 训练 GRU4Rec 模型 =====
    print(f"\n[1] 训练 GRU4Rec + CE")
    torch.manual_seed(42)
    np.random.seed(42)

    model = build_recommendation_model(
        "gru4rec", num_items, embedding_dim=64, hidden_dim=128,
        num_layers=1, dropout=0.2, max_len=max_seq_len
    ).to(device)

    train_dataset = RecDataset(aug_seqs, aug_targets, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_ndcg = 0.0
    best_model_state = None

    for epoch in range(1, 51):
        model.train()
        for seq_batch, length_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(device)
            length_batch = length_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            scores = model(seq_batch, length_batch)
            loss = criterion(scores, target_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        val_ndcg = compute_ndcg_at_k(val_seqs, val_targets, model, device, k=10)
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} | NDCG: {val_ndcg:.4f} | Best: {best_val_ndcg:.4f}")

    print(f"  GRU4Rec 最佳 NDCG: {best_val_ndcg:.4f}")

    # ===== 2. 构建物品协同过滤 =====
    print(f"\n[2] 构建物品协同过滤 (Item-based CF)")

    # 从训练数据构建物品共现矩阵
    item_cooc = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
    for seq, target in zip(train_seqs, train_targets):
        for item in seq:
            if 0 < item <= num_items:
                item_cooc[item][target] += 1
                item_cooc[target][item] += 1

    # 归一化为相似度
    item_norms = np.sqrt((item_cooc ** 2).sum(axis=1, keepdims=True)) + 1e-8
    item_sim = item_cooc / (item_norms * item_norms.T)

    # 计算物品流行度
    item_pop = np.zeros(num_items + 1, dtype=np.float32)
    for target in train_targets:
        item_pop[target] += 1
    item_pop = (item_pop + 1) / (item_pop.sum() + num_items)
    pop_scores = np.log(item_pop + 1e-10)
    pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-10)

    print(f"  物品相似度矩阵: {item_sim.shape}, 非零: {(item_sim > 0).sum()}")

    # ===== 3. 混合预测: GRU4Rec + CF =====
    print(f"\n[3] 混合预测: GRU4Rec + CF")
    model.load_state_dict(best_model_state)
    model.eval()

    test_dataset = TestRecDataset(test_seqs, max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    all_predictions = []
    user_idx = 0

    with torch.no_grad():
        for seq_batch, length_batch in test_loader:
            seq_batch = seq_batch.to(device)
            length_batch = length_batch.to(device)
            model_scores = model(seq_batch, length_batch).cpu().numpy()

            for i in range(model_scores.shape[0]):
                scores = model_scores[i].copy()
                scores[0] = -1e9  # 排除padding

                # 计算CF分数
                seq = test_seqs[user_idx] if user_idx < len(test_seqs) else []
                cf_scores = np.zeros(num_items + 1, dtype=np.float32)
                for item in seq:
                    if 0 < item <= num_items:
                        cf_scores += item_sim[item]

                # 自适应混合权重
                seq_len = len(seq)
                if seq_len == 0:
                    # 无历史: 纯流行度
                    final = pop_scores.copy()
                elif seq_len <= 3:
                    # 极短序列: 40%模型 + 40%CF + 20%流行度
                    final = 0.4 * scores + 0.4 * cf_scores + 0.2 * pop_scores
                elif seq_len <= 10:
                    # 短序列: 60%模型 + 30%CF + 10%流行度
                    final = 0.6 * scores + 0.3 * cf_scores + 0.1 * pop_scores
                else:
                    # 正常序列: 85%模型 + 15%CF
                    final = 0.85 * scores + 0.15 * cf_scores

                final[0] = -1e9
                topk = np.argsort(final)[::-1][:10]
                pred_items = [idx2item[idx] for idx in topk if idx in idx2item and idx > 0]
                while len(pred_items) < 10:
                    pred_items.append(idx2item.get(len(pred_items) + 1, "i000001"))
                all_predictions.append(pred_items[:10])
                user_idx += 1

    elapsed = time.time() - start_time
    print(f"\n[Task2] 完成 | GRU4Rec NDCG = {best_val_ndcg:.4f} | 耗时: {elapsed:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, pred_items in zip(test_uids, all_predictions):
            f.write(f'{uid},"{",".join(pred_items)}"\n')

    return best_val_ndcg


def main():
    print("=" * 70)
    print("  AFAC2026 冲刺0.65: 3轮伪标签+混合集成+CF混合推荐")
    print("=" * 70)

    start_time = time.time()
    cls_score = train_cls_optimized(CLS_DATA_PATH, device="cuda", n_ensemble=20)
    rec_score = train_rec_with_cf(REC_DATA_DIR, device="cuda")

    elapsed = time.time() - start_time
    final = 0.5 * cls_score + 0.5 * rec_score
    print(f"\n{'='*70}")
    print(f"  分类: {cls_score:.4f}")
    print(f"  推荐: {rec_score:.4f}")
    print(f"  总分: {final:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
