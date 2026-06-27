"""
突破版: APPNP分类 + 用户特征融合推荐
分类: GCN+APPNP混合集成 + 伪标签 + 标签传播
推荐: GRU4Rec+用户特征融合 + CE + 序列增强 (解决冷启动)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_classification_model, build_recommendation_model
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.train_cls_improved import drop_edge, label_propagation
from src.train_rec_improved import compute_ndcg_at_k

CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ======================== 分类: GCN+APPNP混合集成 ========================

def run_classification(device="cuda", n_ensemble=20):
    print("\n" + "=" * 70)
    print("  Task1: GCN+APPNP混合集成 + 2轮伪标签 + 标签传播(0.4)")
    print("=" * 70)
    start_time = time.time()

    data = load_classification_data(CLS_DATA_PATH)
    adj = data["adj"]
    features = sk_normalize(data["features"], norm="l2", axis=1).astype(np.float32)
    labels = data["labels"].copy()
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    num_nodes, feat_dim = features.shape
    num_classes = int(labels.max()) + 1

    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")

    # 划分验证集
    tr_sub, val_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=42,
        stratify=labels[train_idx] if len(np.unique(labels[train_idx])) > 1 else None,
    )
    val_idx = train_idx[val_sub]
    train_only = train_idx[tr_sub]

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)
    val_t = torch.LongTensor(val_idx).to(device)
    test_t = torch.LongTensor(test_idx).to(device)

    def train_ensemble(train_indices, n_models):
        """混合架构集成: 50% GCN + 30% APPNP + 20% SAGE"""
        types = ["gcn"] * (n_models // 2) + ["appnp"] * (n_models // 3) + ["sage"] * (n_models - n_models // 2 - n_models // 3)
        train_mask = torch.LongTensor(train_indices).to(device)
        train_labels = labels_t[train_mask]
        all_logits, all_vals = [], []

        for i in range(n_models):
            seed = 42 + i * 10
            torch.manual_seed(seed); np.random.seed(seed)
            mtype = types[i % len(types)]

            model = build_classification_model(mtype, feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
            crit = nn.CrossEntropyLoss()

            best_val, best_logits = 0.0, None
            for epoch in range(1, 301):
                model.train(); opt.zero_grad()
                adj_tr = drop_edge(adj_sparse, 0.2)
                logits = model(features_t, adj_tr)
                loss = crit(logits[train_mask], train_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()

                model.eval()
                with torch.no_grad():
                    logits = model(features_t, adj_sparse)
                    vp = logits[val_t].argmax(1).cpu().numpy()
                    vt = labels[val_idx]
                    va = accuracy_score(vt, vp)
                if va > best_val:
                    best_val = va
                    with torch.no_grad():
                        best_logits = F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()

            all_logits.append(best_logits); all_vals.append(best_val)
            if (i+1) % 5 == 0:
                print(f"  模型 {i+1}/{n_models} ({mtype}) Val={best_val:.4f}")

        return np.mean(all_logits, axis=0), np.mean(all_vals)

    # 轮次1
    print(f"\n[轮次1] {n_ensemble}模型混合集成 (GCN+APPNP+SAGE)")
    pred1, val1 = train_ensemble(train_only, n_ensemble)
    print(f"  Val Acc = {val1:.4f}")

    # 轮次2: 伪标签
    tp1 = pred1.argmax(1); conf1 = pred1.max(1)
    mask1 = conf1 > 0.8
    print(f"\n[轮次2] 伪标签(>0.8): {mask1.sum()}/{len(test_idx)} ({mask1.mean()*100:.1f}%)")
    exp_train = np.concatenate([train_only, test_idx[mask1]])
    exp_labels = labels.copy()
    exp_labels[test_idx[mask1]] = tp1[mask1]
    labels_t = torch.LongTensor(exp_labels).to(device)  # 更新标签
    pred2, val2 = train_ensemble(exp_train, n_ensemble)
    print(f"  Val Acc = {val2:.4f} (变化: {val2-val1:+.4f})")

    # 选择最佳
    best_pred, best_val = (pred2, val2) if val2 > val1 else (pred1, val1)

    # 标签传播
    print(f"\n[后处理] 标签传播(0.4)")
    lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    final = 0.6 * best_pred + 0.4 * lp
    test_pred = final.argmax(1)

    elapsed = time.time() - start_time
    print(f"\n[Task1] 完成 | Val={best_val:.4f} | 耗时={elapsed:.0f}s")
    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return best_val


# ======================== 推荐: GRU4Rec + 用户特征 ========================

class RecDatasetWithUser(Dataset):
    """带用户特征的推荐数据集"""
    def __init__(self, sequences, targets, uids, user_feat_dict, max_len=50):
        self.seqs = sequences
        self.targets = targets
        self.uids = uids
        self.user_feat = user_feat_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) == 0:
            sp = [0] * self.max_len; length = 1
        else:
            length = min(len(seq), self.max_len)
            sp = seq[-self.max_len:] + [0] * (self.max_len - len(seq[-self.max_len:]))
        uf = self.user_feat.get(self.uids[idx], torch.zeros(8))
        return torch.LongTensor(sp), torch.LongTensor([length])[0], torch.LongTensor([self.targets[idx]])[0], uf


class TestRecDatasetWithUser(Dataset):
    def __init__(self, sequences, uids, user_feat_dict, max_len=50):
        self.seqs = sequences; self.uids = uids; self.user_feat = user_feat_dict; self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) == 0:
            sp = [0] * self.max_len; length = 1
        else:
            length = min(len(seq), self.max_len)
            sp = seq[-self.max_len:] + [0] * (self.max_len - len(seq[-self.max_len:]))
        uf = self.user_feat.get(self.uids[idx], torch.zeros(8))
        return torch.LongTensor(sp), torch.LongTensor([length])[0], uf


def run_recommendation(device="cuda"):
    print("\n" + "=" * 70)
    print("  Task2: GRU4Rec+用户特征融合 + CE + 序列增强")
    print("=" * 70)
    start_time = time.time()

    data = load_recommendation_data(REC_DATA_DIR)
    train_df = data["train_df"]; test_df = data["test_df"]
    user_df = data["user_df"]
    item2idx = data["item2idx"]; idx2item = data["idx2item"]
    num_items = data["num_items"]

    max_seq_len = 50
    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_seq_len
    )

    # ===== 加载用户特征 =====
    user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
    user_feat_dims = []
    for col in user_feat_cols:
        user_feat_dims.append(int(user_df[col].max()) + 1)

    # 构建用户特征字典
    user_feat_dict = {}
    for _, row in user_df.iterrows():
        uid = row["uid"]
        feat = torch.LongTensor([int(row[col]) for col in user_feat_cols])
        user_feat_dict[uid] = feat

    print(f"[用户特征] {len(user_feat_cols)}个特征, 维度: {user_feat_dims}")

    # 划分训练/验证
    n_val = int(len(train_seqs) * 0.1)
    np.random.seed(42)
    indices = np.random.permutation(len(train_seqs))
    val_seqs = [train_seqs[i] for i in indices[:n_val]]
    val_targets = [train_targets[i] for i in indices[:n_val]]
    val_uids = [train_df.iloc[indices[:n_val]]["uid"].values[i] for i in range(n_val)]
    tr_seqs = [train_seqs[i] for i in indices[n_val:]]
    tr_targets = [train_targets[i] for i in indices[n_val:]]
    tr_uids = [train_df.iloc[indices[n_val:]]["uid"].values[i] for i in range(len(indices) - n_val)]

    # 序列增强
    aug_seqs, aug_targets, aug_uids = list(tr_seqs), list(tr_targets), list(tr_uids)
    for seq, target, uid in zip(tr_seqs, tr_targets, tr_uids):
        if len(seq) > 5:
            for tl in [5, 10]:
                if len(seq) > tl:
                    aug_seqs.append(seq[-tl:]); aug_targets.append(target); aug_uids.append(uid)
    print(f"[增强] {len(tr_seqs)} → {len(aug_seqs)}")

    # ===== 训练 GRU4Rec + 用户特征 =====
    print(f"\n[训练] GRU4Rec + 用户特征融合 + CE")
    torch.manual_seed(42); np.random.seed(42)

    model = build_recommendation_model(
        "gru4rec", num_items, embedding_dim=64, hidden_dim=128,
        num_layers=1, dropout=0.2, max_len=max_seq_len,
        user_feat_dims=user_feat_dims,
    ).to(device)

    train_ds = RecDatasetWithUser(aug_seqs, aug_targets, aug_uids, user_feat_dict, max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    best_ndcg, best_state = 0.0, None
    for epoch in range(1, 51):
        model.train()
        for seq_b, len_b, tgt_b, uf_b in train_loader:
            seq_b = seq_b.to(device); len_b = len_b.to(device)
            tgt_b = tgt_b.to(device); uf_b = uf_b.to(device)
            opt.zero_grad()
            scores = model(seq_b, len_b, uf_b)
            loss = crit(scores, tgt_b.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()

        # 验证
        model.eval()
        ndcgs = []
        with torch.no_grad():
            for start in range(0, len(val_seqs), 256):
                batch_seqs = val_seqs[start:start+256]
                batch_targets = val_targets[start:start+256]
                batch_uids = val_uids[start:start+256]
                seqs, lens, ufs = [], [], []
                for seq, uid in zip(batch_seqs, batch_uids):
                    if len(seq) == 0:
                        sp = [0]*max_seq_len; length = 1
                    else:
                        length = min(len(seq), max_seq_len)
                        sp = seq[-max_seq_len:] + [0]*(max_seq_len-len(seq[-max_seq_len:]))
                    seqs.append(sp); lens.append(length)
                    ufs.append(user_feat_dict.get(uid, torch.zeros(8)))
                seq_t = torch.LongTensor(seqs).to(device)
                len_t = torch.LongTensor(lens).to(device)
                uf_t = torch.stack(ufs).to(device)
                scores = model(seq_t, len_t, uf_t)
                scores[:, 0] = -1e9
                _, topk = scores.topk(10, dim=1)
                topk = topk.cpu().numpy()
                for i, tgt in enumerate(batch_targets):
                    if tgt in topk[i]:
                        rank = np.where(topk[i] == tgt)[0][0]
                        ndcgs.append(1.0 / np.log2(rank + 2))
                    else:
                        ndcgs.append(0.0)
        val_ndcg = float(np.mean(ndcgs))

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} | NDCG: {val_ndcg:.4f} | Best: {best_ndcg:.4f}")

    print(f"  最佳 NDCG: {best_ndcg:.4f}")

    # ===== 预测 =====
    model.load_state_dict(best_state); model.eval()
    test_ds = TestRecDatasetWithUser(test_seqs, test_uids, user_feat_dict, max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for seq_b, len_b, uf_b in test_loader:
            seq_b = seq_b.to(device); len_b = len_b.to(device); uf_b = uf_b.to(device)
            scores = model(seq_b, len_b, uf_b)
            scores[:, 0] = -1e9
            _, topk = scores.topk(10, dim=1)
            topk = topk.cpu().numpy()
            for pred in topk:
                items = [idx2item[i] for i in pred if i in idx2item and i > 0]
                while len(items) < 10:
                    items.append(idx2item.get(len(items)+1, "i000001"))
                all_preds.append(items[:10])

    elapsed = time.time() - start_time
    print(f"\n[Task2] 完成 | NDCG={best_ndcg:.4f} | 耗时={elapsed:.0f}s")
    with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, items in zip(test_uids, all_preds):
            f.write(f'{uid},"{",".join(items)}"\n')
    return best_ndcg


def main():
    print("=" * 70)
    print("  AFAC2026 突破版: APPNP+用户特征融合")
    print("=" * 70)
    t0 = time.time()
    cls = run_classification(device="cuda", n_ensemble=20)
    rec = run_recommendation(device="cuda")
    elapsed = time.time() - t0
    final = 0.5 * cls + 0.5 * rec
    print(f"\n{'='*70}")
    print(f"  分类: {cls:.4f}")
    print(f"  推荐: {rec:.4f}")
    print(f"  总分: {final:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
