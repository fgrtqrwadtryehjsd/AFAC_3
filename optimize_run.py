"""
优化版运行脚本 - 冲刺0.65分
分类: GCN + DropEdge + 20集成 + 伪标签 + 标签传播(0.4)
推荐: SASRec + CE + 3集成 + 滑动窗口增强
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize
import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_classification_model, build_recommendation_model
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.train_cls_improved import (
    drop_edge, normalize_features, label_propagation, build_knn_graph
)
from src.train_rec_improved import (
    RecDatasetBPR, TestRecDataset, compute_ndcg_at_k, compute_rec_diagnostic_report
)
from torch.utils.data import DataLoader

CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ======================== 分类任务优化 ========================

def train_cls_with_pseudo_labels(npz_path, device="cuda", n_ensemble=20):
    """分类任务: GCN + DropEdge + 集成 + 伪标签 + 标签传播"""

    print("\n" + "=" * 70)
    print("  Task1-Optimized: GCN + DropEdge + 20集成 + 伪标签 + 标签传播(0.4)")
    print("=" * 70)

    start_time = time.time()

    # 加载数据
    data = load_classification_data(npz_path)
    adj = data["adj"]
    features = data["features"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    num_nodes, feat_dim = features.shape
    num_classes = int(labels.max()) + 1

    # L2 特征归一化
    features = sk_normalize(features, norm="l2", axis=1).astype(np.float32)

    # 邻接矩阵预处理 (对称归一化 + 自环)
    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")

    # 划分训练/验证集
    train_labels_arr = labels[train_idx]
    train_sub, val_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=42,
        stratify=train_labels_arr if len(np.unique(train_labels_arr)) > 1 else None,
    )
    val_idx = train_idx[val_sub]
    train_only_idx = train_idx[train_sub]

    # ===== 阶段1: 第一轮集成训练 (DropEdge + 标签传播) =====
    print(f"\n[阶段1] 第一轮集成训练 ({n_ensemble} 模型, DropEdge=0.2)")

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)
    train_mask_t = torch.LongTensor(train_only_idx).to(device)
    val_mask_t = torch.LongTensor(val_idx).to(device)
    test_idx_t = torch.LongTensor(test_idx).to(device)

    ensemble_logits_1 = []
    val_accs_1 = []

    for i in range(n_ensemble):
        seed = 42 + i * 10
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = build_classification_model(
            "gcn", feat_dim, 256, num_classes, 2, 0.5, "sym"
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
            loss = criterion(logits[train_mask_t], labels_t[train_mask_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                logits = model(features_t, adj_sparse)
                val_pred = logits[val_mask_t].argmax(dim=1).cpu().numpy()
                val_true = labels_t[val_mask_t].cpu().numpy()
                val_acc = accuracy_score(val_true, val_pred)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                with torch.no_grad():
                    best_test_logits = F.softmax(
                        model(features_t, adj_sparse)[test_idx_t], dim=1
                    ).cpu().numpy()

            if epoch % 50 == 0:
                print(f"    [Seed {seed}] Epoch {epoch} | Loss: {loss.item():.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        ensemble_logits_1.append(best_test_logits)
        val_accs_1.append(best_val_acc)
        if (i + 1) % 5 == 0:
            print(f"  模型 {i+1}/{n_ensemble} 完成, Val Acc = {best_val_acc:.4f}")

    avg_val_acc_1 = np.mean(val_accs_1)
    print(f"\n[阶段1] 集成平均 Val Acc = {avg_val_acc_1:.4f}")

    # 集成预测
    ensemble_pred_1 = np.mean(ensemble_logits_1, axis=0)

    # ===== 阶段2: 伪标签 (Pseudo-labeling) =====
    print(f"\n[阶段2] 伪标签: 用高置信度测试预测扩充训练集")

    # 获取测试集伪标签 (置信度 > 0.8)
    test_pred_labels = ensemble_pred_1.argmax(axis=1)
    test_confidence = ensemble_pred_1.max(axis=1)
    high_conf_mask = test_confidence > 0.8
    pseudo_train_idx = test_idx[high_conf_mask]
    pseudo_train_labels = test_pred_labels[high_conf_mask]

    print(f"  高置信度伪标签: {high_conf_mask.sum()}/{len(test_idx)} ({high_conf_mask.mean()*100:.1f}%)")

    # 扩展训练集
    expanded_train_idx = np.concatenate([train_idx, pseudo_train_idx])
    expanded_labels = labels.copy()
    expanded_labels[pseudo_train_idx] = pseudo_train_labels

    expanded_train_mask_t = torch.LongTensor(expanded_train_idx).to(device)
    expanded_labels_t = torch.LongTensor(expanded_labels).to(device)

    # ===== 阶段3: 第二轮集成训练 (用扩展训练集) =====
    print(f"\n[阶段3] 第二轮集成训练 (伪标签扩充, {n_ensemble} 模型)")

    ensemble_logits_2 = []
    val_accs_2 = []

    for i in range(n_ensemble):
        seed = 200 + i * 10
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = build_classification_model(
            "gcn", feat_dim, 256, num_classes, 2, 0.5, "sym"
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
            loss = criterion(logits[expanded_train_mask_t], expanded_labels_t[expanded_train_mask_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                logits = model(features_t, adj_sparse)
                val_pred = logits[val_mask_t].argmax(dim=1).cpu().numpy()
                val_true = labels_t[val_mask_t].cpu().numpy()
                val_acc = accuracy_score(val_true, val_pred)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                with torch.no_grad():
                    best_test_logits = F.softmax(
                        model(features_t, adj_sparse)[test_idx_t], dim=1
                    ).cpu().numpy()

        ensemble_logits_2.append(best_test_logits)
        val_accs_2.append(best_val_acc)

    avg_val_acc_2 = np.mean(val_accs_2)
    print(f"\n[阶段3] 伪标签后集成 Val Acc = {avg_val_acc_2:.4f} (第一轮: {avg_val_acc_1:.4f})")

    # 选择更好的结果
    if avg_val_acc_2 > avg_val_acc_1:
        ensemble_pred = np.mean(ensemble_logits_2, axis=0)
        best_val_acc = avg_val_acc_2
        print(f"[选择] 使用伪标签结果 (提升 {avg_val_acc_2 - avg_val_acc_1:.4f})")
    else:
        ensemble_pred = ensemble_pred_1
        best_val_acc = avg_val_acc_1
        print(f"[选择] 使用第一轮结果 (伪标签未提升)")

    # ===== 阶段4: 标签传播后处理 =====
    print(f"\n[阶段4] 标签传播后处理 (权重=0.4)")
    lp_logits = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp_logits = lp_logits / (lp_logits.sum(axis=1, keepdims=True) + 1e-8)
    final_logits = 0.6 * ensemble_pred + 0.4 * lp_logits
    test_pred = final_logits.argmax(axis=1)

    elapsed = time.time() - start_time
    print(f"\n[Task1] 完成 | Val Acc = {best_val_acc:.4f} | 耗时: {elapsed:.1f}s")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, pred in zip(test_idx, test_pred):
            f.write(f"{idx},{pred}\n")

    return best_val_acc


# ======================== 推荐任务优化 ========================

def train_rec_optimized(data_dir, device="cuda", n_ensemble=3):
    """推荐任务: SASRec + CE + 3集成 + 滑动窗口增强"""

    print("\n" + "=" * 70)
    print("  Task2-Optimized: SASRec + CE + 3集成 + 滑动窗口增强")
    print("=" * 70)

    start_time = time.time()

    # 加载数据
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

    # 滑动窗口增强
    aug_seqs, aug_targets = [], []
    for seq, target in zip(tr_seqs, tr_targets):
        aug_seqs.append(seq)
        aug_targets.append(target)
        # 滑动窗口: 从长序列中截取多个子序列
        if len(seq) > 10:
            for start in range(0, len(seq) - 5, 5):
                aug_seqs.append(seq[start:start + 10])
                aug_targets.append(target)

    print(f"[增强] 训练样本: {len(tr_seqs)} → {len(aug_seqs)} (滑动窗口 +{len(aug_seqs)-len(tr_seqs)})")

    # 集成训练
    all_test_probs = []
    val_ndcgs = []

    for model_idx in range(n_ensemble):
        seed = 42 + model_idx * 100
        print(f"\n--- 模型 {model_idx+1}/{n_ensemble} (seed={seed}) ---")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # 使用 SASRec
        model = build_recommendation_model(
            "sasrec", num_items, embedding_dim=64, hidden_dim=128,
            num_layers=2, dropout=0.2, max_len=max_seq_len
        ).to(device)

        # CE 损失
        from src.train_rec import RecDataset
        train_dataset = RecDataset(aug_seqs, aug_targets, max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_val_ndcg = 0.0
        best_model_state = None

        for epoch in range(1, 51):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

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

                epoch_loss += loss.item()
                n_batches += 1

            # 验证
            val_ndcg = compute_ndcg_at_k(val_seqs, val_targets, model, device, k=10, batch_size=256)

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch} | Loss: {epoch_loss/n_batches:.4f} | NDCG: {val_ndcg:.4f} | Best: {best_val_ndcg:.4f}")

        val_ndcgs.append(best_val_ndcg)
        print(f"  模型 {model_idx+1} 最佳 NDCG: {best_val_ndcg:.4f}")

        # 用最佳模型预测测试集
        model.load_state_dict(best_model_state)
        model.eval()

        test_dataset = TestRecDataset(test_seqs, max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

        test_probs = []
        with torch.no_grad():
            for seq_batch, length_batch in test_loader:
                seq_batch = seq_batch.to(device)
                length_batch = length_batch.to(device)
                scores = model(seq_batch, length_batch)
                probs = F.softmax(scores, dim=1).cpu().numpy()
                test_probs.append(probs)

        all_test_probs.append(np.concatenate(test_probs, axis=0))

    # 集成预测 (平均概率)
    avg_val_ndcg = np.mean(val_ndcgs)
    ensemble_probs = np.mean(all_test_probs, axis=0)

    elapsed = time.time() - start_time
    print(f"\n[Task2] 完成 | 集成 NDCG = {avg_val_ndcg:.4f} | 耗时: {elapsed:.1f}s")

    # 生成预测
    all_predictions = []
    for i in range(ensemble_probs.shape[0]):
        probs = ensemble_probs[i].copy()
        probs[0] = -1e9  # 排除 padding
        topk_indices = np.argsort(probs)[::-1][:10]
        pred_items = [idx2item[idx] for idx in topk_indices if idx in idx2item and idx > 0]
        while len(pred_items) < 10:
            pred_items.append(idx2item.get(len(pred_items) + 1, "i000001"))
        all_predictions.append(pred_items[:10])

    with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, pred_items in zip(test_uids, all_predictions):
            f.write(f'{uid},"{",".join(pred_items)}"\n')

    return avg_val_ndcg


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 优化版冲刺")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="both", choices=["both", "cls", "rec"])
    parser.add_argument("--n_ensemble_cls", type=int, default=20)
    parser.add_argument("--n_ensemble_rec", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 优化版冲刺 0.65")
    print("=" * 70)

    start_time = time.time()
    cls_score = 0
    rec_score = 0

    if args.task in ("both", "cls"):
        cls_score = train_cls_with_pseudo_labels(
            CLS_DATA_PATH, device=args.device, n_ensemble=args.n_ensemble_cls
        )

    if args.task in ("both", "rec"):
        rec_score = train_rec_optimized(
            REC_DATA_DIR, device=args.device, n_ensemble=args.n_ensemble_rec
        )

    elapsed = time.time() - start_time
    final = 0.5 * cls_score + 0.5 * rec_score

    print("\n" + "=" * 70)
    print("  最终结果")
    print("=" * 70)
    print(f"  分类: {cls_score:.4f}")
    print(f"  推荐: {rec_score:.4f}")
    print(f"  总分: {final:.4f}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
