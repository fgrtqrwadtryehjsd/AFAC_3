"""
突破版V2: 2跳邻接+PCA分类 + 短序列训练推荐
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import PCA
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


def run_classification(device="cuda", n_ensemble=20):
    print("\n" + "=" * 70)
    print("  Task1: 2跳邻接+PCA+GCN集成+伪标签+标签传播")
    print("=" * 70)
    start_time = time.time()

    data = load_classification_data(CLS_DATA_PATH)
    adj = data["adj"]
    features = data["features"]
    labels = data["labels"].copy()
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    num_nodes, feat_dim = features.shape
    num_classes = int(labels.max()) + 1
    degree = np.array(adj.sum(axis=1)).flatten()

    # ===== PCA降维去噪 =====
    print(f"[PCA] 原始维度: {feat_dim}")
    pca = PCA(n_components=min(256, feat_dim), random_state=42)
    features_pca = pca.fit_transform(features).astype(np.float32)
    features_pca = sk_normalize(features_pca, norm="l2", axis=1).astype(np.float32)
    print(f"[PCA] 降维后: {features_pca.shape[1]}, 解释方差: {pca.explained_variance_ratio_.sum():.3f}")

    # ===== 2跳邻接增强 =====
    print(f"[2跳邻接] 原始边数: {adj.nnz}")
    adj_2hop = (adj @ adj).tocoo()
    # 过滤过大的2跳值
    adj_2hop_data = np.ones(len(adj_2hop.data))
    adj_2hop = sp.csr_matrix((adj_2hop_data, (adj_2hop.row, adj_2hop.col)), shape=adj.shape)
    adj_enhanced = (adj + adj_2hop).tocoo()
    adj_enhanced_data = np.ones(len(adj_enhanced.data))
    adj_enhanced = sp.csr_matrix((adj_enhanced_data, (adj_enhanced.row, adj_enhanced.col)), shape=adj.shape)
    print(f"[2跳邻接] 增强后边数: {adj_enhanced.nnz} (1跳: {adj.nnz}, 2跳: {adj_2hop.nnz})")

    # 归一化
    adj_norm = preprocess_adj(adj_enhanced, add_self_loops=True, normalization="sym")

    # 划分验证集
    tr_sub, val_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=42,
        stratify=labels[train_idx] if len(np.unique(labels[train_idx])) > 1 else None)
    val_idx = train_idx[val_sub]
    train_only = train_idx[tr_sub]

    features_t = torch.FloatTensor(features_pca).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)
    val_t = torch.LongTensor(val_idx).to(device)
    test_t = torch.LongTensor(test_idx).to(device)

    def train_ensemble(train_indices, train_labels_arr, n_models):
        train_mask = torch.LongTensor(train_indices).to(device)
        train_labels = torch.LongTensor(train_labels_arr).to(device)
        all_logits, all_vals = [], []
        for i in range(n_models):
            seed = 42 + i * 10
            torch.manual_seed(seed); np.random.seed(seed)
            model = build_classification_model("gcn", features_pca.shape[1], 256, num_classes, 2, 0.5, "sym").to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
            best_val, best_logits = 0.0, None
            for epoch in range(1, 301):
                model.train(); opt.zero_grad()
                adj_tr = drop_edge(adj_sparse, 0.2)
                logits = model(features_t, adj_tr)
                loss = crit(logits[train_mask], train_labels[train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
                model.eval()
                with torch.no_grad():
                    logits = model(features_t, adj_sparse)
                    vp = logits[val_t].argmax(1).cpu().numpy()
                    va = accuracy_score(labels[val_idx], vp)
                if va > best_val:
                    best_val = va
                    with torch.no_grad():
                        best_logits = F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
            all_logits.append(best_logits); all_vals.append(best_val)
            if (i+1) % 5 == 0:
                print(f"  模型 {i+1}/{n_models} Val={best_val:.4f}")
        return np.mean(all_logits, axis=0), np.mean(all_vals)

    # 轮次1: 基线
    print(f"\n[轮次1] {n_ensemble}模型集成 (2跳邻接+PCA+标签平滑)")
    pred1, val1 = train_ensemble(train_only, labels, n_ensemble)
    print(f"  Val Acc = {val1:.4f}")

    # 轮次2: 伪标签
    tp1 = pred1.argmax(1); conf1 = pred1.max(1)
    mask1 = conf1 > 0.8
    print(f"\n[轮次2] 伪标签(>0.8): {mask1.sum()}/{len(test_idx)}")
    exp_train = np.concatenate([train_only, test_idx[mask1]])
    exp_labels = labels.copy()
    exp_labels[test_idx[mask1]] = tp1[mask1]
    labels_t = torch.LongTensor(exp_labels).to(device)
    pred2, val2 = train_ensemble(exp_train, exp_labels, n_ensemble)
    print(f"  Val Acc = {val2:.4f} (变化: {val2-val1:+.4f})")

    # 轮次3: 伪标签递减
    best_pred = pred2 if val2 > val1 else pred1
    tp2 = best_pred.argmax(1); conf2 = best_pred.max(1)
    mask2 = conf2 > 0.7
    print(f"\n[轮次3] 伪标签(>0.7): {mask2.sum()}/{len(test_idx)}")
    exp_train2 = np.concatenate([train_only, test_idx[mask2]])
    exp_labels2 = labels.copy()
    exp_labels2[test_idx[mask2]] = tp2[mask2]
    labels_t = torch.LongTensor(exp_labels2).to(device)
    pred3, val3 = train_ensemble(exp_train2, exp_labels2, n_ensemble)
    print(f"  Val Acc = {val3:.4f} (变化: {val3-val2:+.4f})")

    # 选择最佳
    results = [(pred1, val1, "R1"), (pred2, val2, "R2"), (pred3, val3, "R3")]
    best_pred, best_val, best_name = max(results, key=lambda x: x[1])
    print(f"\n[选择] {best_name} Val={best_val:.4f}")

    # 最终重训: 100%数据+伪标签
    print(f"\n[最终重训] 100%训练数据+伪标签")
    tp_best = best_pred.argmax(1); conf_best = best_pred.max(1)
    mask_final = conf_best > 0.7
    final_train = np.concatenate([train_idx, test_idx[mask_final]])
    final_labels = labels.copy()
    final_labels[test_idx[mask_final]] = tp_best[mask_final]
    pred_final, val_final = train_ensemble(final_train, final_labels, n_ensemble)
    print(f"  最终重训 Val Acc = {val_final:.4f}")

    # 选择最终最佳
    if val_final > best_val:
        best_pred = pred_final
        print(f"[选择] 使用最终重训结果")

    # 标签传播
    print(f"\n[标签传播] 权重=0.4")
    lp = label_propagation(features_pca, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    final_pred = 0.6 * best_pred + 0.4 * lp
    test_pred = final_pred.argmax(1)

    elapsed = time.time() - start_time
    print(f"\n[Task1] 完成 | Val={best_val:.4f} | 耗时={elapsed:.0f}s")
    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")

    # 保存轨迹
    traj = {
        "task_type": "classification",
        "total_rounds": 4,
        "experiments": [
            {"round": 1, "config": {"model": "gcn", "features": "PCA(256)+L2", "graph": "2-hop adjacency", "label_smoothing": 0.1, "n_ensemble": n_ensemble, "drop_edge": 0.2},
             "val_accuracy": float(val1), "rationale": "2跳邻接解决53%孤立节点问题, PCA降维去噪, 标签平滑正则化"},
            {"round": 2, "config": {"use_pseudo_label": True, "threshold": 0.8},
             "val_accuracy": float(val2), "rationale": "SOP触发: 准确率>0.65→伪标签"},
            {"round": 3, "config": {"use_pseudo_label": True, "threshold": 0.7},
             "val_accuracy": float(val3), "rationale": "伪标签递减阈值"},
            {"round": "final", "config": {"train_data": "100%+pseudo", "threshold": 0.7},
             "val_accuracy": float(val_final), "rationale": "最终重训: 用全部训练数据+伪标签"},
        ],
        "best_result": {"val_accuracy": float(max(val1, val2, val3, val_final))},
        "techniques": ["2-hop adjacency (A+A²)", "PCA(767→256)", "label_smoothing(0.1)", "DropEdge(0.2)", "pseudo-labeling(3 rounds)", "label_propagation(0.4)"],
    }
    with open(os.path.join(OUTPUT_DIR, "trajectory_B1.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)

    return max(val1, val2, val3, val_final)


def run_recommendation(device="cuda"):
    print("\n" + "=" * 70)
    print("  Task2: 短序列训练+用户特征+物品特征+CE")
    print("=" * 70)
    start_time = time.time()

    data = load_recommendation_data(REC_DATA_DIR)
    train_df = data["train_df"]; test_df = data["test_df"]
    user_df = data["user_df"]
    item2idx = data["item2idx"]; idx2item = data["idx2item"]
    num_items = data["num_items"]

    # 短序列长度: 匹配测试分布(均值6.25)
    max_seq_len = 15
    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_seq_len)
    print(f"[配置] max_seq_len={max_seq_len} (测试均值6.25)")

    # 用户特征
    user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
    user_feat_dims = [int(user_df[col].max()) + 1 for col in user_feat_cols]
    user_feat_dict = {}
    for _, row in user_df.iterrows():
        user_feat_dict[row["uid"]] = torch.LongTensor([int(row[c]) for c in user_feat_cols])

    # 划分: 用短序列用户做验证 (匹配测试分布)
    seq_lens = [len(s) for s in train_seqs]
    short_mask = np.array(seq_lens) <= 10  # 短序列用户做验证
    short_indices = np.where(short_mask)[0]
    long_indices = np.where(~short_mask)[0]
    np.random.seed(42)
    np.random.shuffle(short_indices)
    n_val = min(len(short_indices) // 3, 2000)  # 从短序列用户中选验证集
    val_indices = short_indices[:n_val]
    tr_indices = np.concatenate([long_indices, short_indices[n_val:]])

    val_seqs = [train_seqs[i] for i in val_indices]
    val_targets = [train_targets[i] for i in val_indices]
    val_uids = [train_df.iloc[val_indices]["uid"].values[i] for i in range(len(val_indices))]
    tr_seqs = [train_seqs[i] for i in tr_indices]
    tr_targets = [train_targets[i] for i in tr_indices]
    tr_uids = [train_df.iloc[tr_indices]["uid"].values[i] for i in range(len(tr_indices))]

    print(f"[验证集] {len(val_seqs)}个短序列用户(匹配测试分布), 训练集{len(tr_seqs)}")

    # 序列增强: 更多短序列
    aug_seqs, aug_targets, aug_uids = list(tr_seqs), list(tr_targets), list(tr_uids)
    for seq, tgt, uid in zip(tr_seqs, tr_targets, tr_uids):
        if len(seq) > 3:
            for tl in [3, 5, 8, 10]:
                if len(seq) > tl:
                    aug_seqs.append(seq[-tl:]); aug_targets.append(tgt); aug_uids.append(uid)
    print(f"[增强] {len(tr_seqs)} → {len(aug_seqs)}")

    # 训练
    print(f"\n[训练] GRU4Rec+用户特征+CE (max_seq_len={max_seq_len})")
    torch.manual_seed(42); np.random.seed(42)
    model = build_recommendation_model(
        "gru4rec", num_items, embedding_dim=64, hidden_dim=128,
        num_layers=1, dropout=0.2, max_len=max_seq_len,
        user_feat_dims=user_feat_dims).to(device)

    class RecDS(torch.utils.data.Dataset):
        def __init__(self, s, t, u, uf, ml):
            self.s, self.t, self.u, self.uf, self.ml = s, t, u, uf, ml
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            seq = self.s[i]
            if not seq: sp = [0]*self.ml; length = 1
            else:
                length = min(len(seq), self.ml)
                sp = seq[-self.ml:] + [0]*(self.ml-len(seq[-self.ml:]))
            return torch.LongTensor(sp), torch.LongTensor([length])[0], torch.LongTensor([self.t[i]])[0], self.uf.get(self.u[i], torch.zeros(8))

    train_ds = RecDS(aug_seqs, aug_targets, aug_uids, user_feat_dict, max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    best_ndcg, best_state = 0.0, None
    for epoch in range(1, 51):
        model.train()
        for sb, lb, tb, uf in train_loader:
            sb, lb, tb, uf = sb.to(device), lb.to(device), tb.to(device), uf.to(device)
            opt.zero_grad()
            scores = model(sb, lb, uf)
            loss = crit(scores, tb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()

        # 验证(短序列用户)
        model.eval()
        ndcgs = []
        with torch.no_grad():
            for start in range(0, len(val_seqs), 256):
                bs, bt, bu = val_seqs[start:start+256], val_targets[start:start+256], val_uids[start:start+256]
                seqs, lens, ufs = [], [], []
                for seq, uid in zip(bs, bu):
                    if not seq: sp = [0]*max_seq_len; length = 1
                    else:
                        length = min(len(seq), max_seq_len)
                        sp = seq[-max_seq_len:] + [0]*(max_seq_len-len(seq[-max_seq_len:]))
                    seqs.append(sp); lens.append(length)
                    ufs.append(user_feat_dict.get(uid, torch.zeros(8)))
                st = torch.LongTensor(seqs).to(device)
                lt = torch.LongTensor(lens).to(device)
                uft = torch.stack(ufs).to(device)
                scores = model(st, lt, uft)
                scores[:, 0] = -1e9
                _, topk = scores.topk(10, dim=1)
                topk = topk.cpu().numpy()
                for i, tgt in enumerate(bt):
                    if tgt in topk[i]:
                        ndcgs.append(1.0 / np.log2(np.where(topk[i]==tgt)[0][0] + 2))
                    else:
                        ndcgs.append(0.0)
        val_ndcg = float(np.mean(ndcgs))
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} | NDCG: {val_ndcg:.4f} | Best: {best_ndcg:.4f}")

    print(f"  最佳NDCG(短序列验证): {best_ndcg:.4f}")

    # 预测
    model.load_state_dict(best_state); model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_seqs), 256):
            bs, bu = test_seqs[start:start+256], test_uids[start:start+256]
            seqs, lens, ufs = [], [], []
            for seq, uid in zip(bs, bu):
                if not seq: sp = [0]*max_seq_len; length = 1
                else:
                    length = min(len(seq), max_seq_len)
                    sp = seq[-max_seq_len:] + [0]*(max_seq_len-len(seq[-max_seq_len:]))
                seqs.append(sp); lens.append(length)
                ufs.append(user_feat_dict.get(uid, torch.zeros(8)))
            st = torch.LongTensor(seqs).to(device)
            lt = torch.LongTensor(lens).to(device)
            uft = torch.stack(ufs).to(device)
            scores = model(st, lt, uft)
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

    # 轨迹
    traj = {
        "task_type": "recommendation",
        "total_rounds": 1,
        "experiments": [
            {"round": 1, "config": {"model": "gru4rec", "loss": "ce", "max_seq_len": max_seq_len,
              "user_features": True, "seq_aug": True, "val_strategy": "short_seq_users"},
             "val_ndcg": float(best_ndcg),
             "rationale": "短序列训练(max_seq_len=15)匹配测试分布(均值6.25), 短序列验证集做模型选择"},
        ],
        "best_result": {"val_ndcg": float(best_ndcg)},
        "techniques": ["short_seq_training(15)", "short_seq_validation", "user_features", "seq_augmentation(trunc 3/5/8/10)"],
    }
    with open(os.path.join(OUTPUT_DIR, "trajectory_B2.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)

    return best_ndcg


def main():
    print("=" * 70)
    print("  AFAC2026 突破V2: 2跳邻接+PCA+短序列训练")
    print("=" * 70)
    t0 = time.time()
    cls = run_classification(device="cuda", n_ensemble=20)
    rec = run_recommendation(device="cuda")
    elapsed = time.time() - t0
    final = 0.5 * cls + 0.5 * rec
    print(f"\n{'='*70}")
    print(f"  分类: {cls:.4f} | 推荐: {rec:.4f} | 总分: {final:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
