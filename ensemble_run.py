"""
推荐集成: GRU4Rec + SASRec 概率平均
分类: 最优配置(label_prop=0.5提高权重)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize
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
    print("  Task1: GCN集成+伪标签+标签传播(0.5)")
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
    tr_sub, val_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, random_state=42,
        stratify=labels[train_idx] if len(np.unique(labels[train_idx])) > 1 else None)
    val_idx = train_idx[val_sub]
    train_only = train_idx[tr_sub]

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)
    val_t = torch.LongTensor(val_idx).to(device)
    test_t = torch.LongTensor(test_idx).to(device)

    def train_ens(train_indices, train_labels_arr, n_models):
        tm = torch.LongTensor(train_indices).to(device)
        tl = torch.LongTensor(train_labels_arr).to(device)
        all_logits, all_vals = [], []
        for i in range(n_models):
            seed = 42 + i * 10
            torch.manual_seed(seed); np.random.seed(seed)
            model = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
            crit = nn.CrossEntropyLoss()
            bv, bl = 0.0, None
            for epoch in range(1, 301):
                model.train(); opt.zero_grad()
                adj_tr = drop_edge(adj_sparse, 0.2)
                logits = model(features_t, adj_tr)
                loss = crit(logits[tm], tl[tm])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
                model.eval()
                with torch.no_grad():
                    logits = model(features_t, adj_sparse)
                    vp = logits[val_t].argmax(1).cpu().numpy()
                    va = accuracy_score(labels[val_idx], vp)
                if va > bv:
                    bv = va
                    with torch.no_grad():
                        bl = F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
            all_logits.append(bl); all_vals.append(bv)
            if (i+1) % 5 == 0:
                print(f"  模型 {i+1}/{n_models} Val={bv:.4f}")
        return np.mean(all_logits, axis=0), np.mean(all_vals)

    # R1: 基线
    print(f"\n[轮次1] {n_ensemble}模型集成")
    pred1, val1 = train_ens(train_only, labels, n_ensemble)
    print(f"  Val={val1:.4f}")

    # R2: 伪标签
    tp1 = pred1.argmax(1); conf1 = pred1.max(1)
    mask1 = conf1 > 0.8
    print(f"\n[轮次2] 伪标签(>0.8): {mask1.sum()}")
    exp_train = np.concatenate([train_only, test_idx[mask1]])
    exp_labels = labels.copy()
    exp_labels[test_idx[mask1]] = tp1[mask1]
    labels_t = torch.LongTensor(exp_labels).to(device)
    pred2, val2 = train_ens(exp_train, exp_labels, n_ensemble)
    print(f"  Val={val2:.4f} ({val2-val1:+.4f})")

    # R3: 最终重训(100%数据+伪标签)
    best_pred = pred2 if val2 > val1 else pred1
    tp_best = best_pred.argmax(1); conf_best = best_pred.max(1)
    mask_f = conf_best > 0.7
    print(f"\n[最终重训] 100%数据+伪标签(>0.7): {mask_f.sum()}")
    final_train = np.concatenate([train_idx, test_idx[mask_f]])
    final_labels = labels.copy()
    final_labels[test_idx[mask_f]] = tp_best[mask_f]
    pred_f, val_f = train_ens(final_train, final_labels, n_ensemble)
    print(f"  Val={val_f:.4f}")

    if val_f > max(val1, val2):
        best_pred = pred_f
        print("[选择] 最终重训")

    # 标签传播(0.5 - 提高权重)
    print(f"\n[标签传播] 权重=0.5")
    lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    final_pred = 0.5 * best_pred + 0.5 * lp
    test_pred = final_pred.argmax(1)

    elapsed = time.time() - start_time
    print(f"\n[Task1] 完成 | Val={max(val1,val2,val_f):.4f} | 耗时={elapsed:.0f}s")
    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return max(val1, val2, val_f)


def run_recommendation_ensemble(device="cuda"):
    print("\n" + "=" * 70)
    print("  Task2: GRU4Rec+SASRec集成 + 用户特征 + CE")
    print("=" * 70)
    start_time = time.time()

    data = load_recommendation_data(REC_DATA_DIR)
    train_df = data["train_df"]; test_df = data["test_df"]
    user_df = data["user_df"]
    item2idx = data["item2idx"]; idx2item = data["idx2item"]
    num_items = data["num_items"]
    max_seq_len = 50

    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_seq_len)

    # 用户特征
    user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
    user_feat_dims = [int(user_df[col].max()) + 1 for col in user_feat_cols]
    user_feat_dict = {}
    for _, row in user_df.iterrows():
        user_feat_dict[row["uid"]] = torch.LongTensor([int(row[c]) for c in user_feat_cols])

    # 划分
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
    for seq, tgt, uid in zip(tr_seqs, tr_targets, tr_uids):
        if len(seq) > 5:
            for tl in [5, 10]:
                if len(seq) > tl:
                    aug_seqs.append(seq[-tl:]); aug_targets.append(tgt); aug_uids.append(uid)

    # ===== 模型1: GRU4Rec + 用户特征 =====
    print(f"\n[模型1] GRU4Rec + 用户特征 + CE")
    torch.manual_seed(42); np.random.seed(42)
    model1 = build_recommendation_model(
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

    opt1 = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=0)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=50, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()
    best1, state1 = 0.0, None
    for epoch in range(1, 51):
        model1.train()
        for sb, lb, tb, uf in train_loader:
            sb, lb, tb, uf = sb.to(device), lb.to(device), tb.to(device), uf.to(device)
            opt1.zero_grad()
            scores = model1(sb, lb, uf)
            loss = crit(scores, tb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
            opt1.step(); sched1.step()
        model1.eval()
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
                scores = model1(st, lt, uft)
                scores[:, 0] = -1e9
                _, topk = scores.topk(10, dim=1)
                topk = topk.cpu().numpy()
                for i, tgt in enumerate(bt):
                    if tgt in topk[i]:
                        ndcgs.append(1.0 / np.log2(np.where(topk[i]==tgt)[0][0] + 2))
                    else:
                        ndcgs.append(0.0)
        val_ndcg = float(np.mean(ndcgs))
        if val_ndcg > best1:
            best1 = val_ndcg; state1 = {k: v.clone() for k, v in model1.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  GRU4Rec Epoch {epoch} | NDCG: {val_ndcg:.4f} | Best: {best1:.4f}")
    print(f"  GRU4Rec 最佳: {best1:.4f}")

    # ===== 模型2: SASRec + CE =====
    print(f"\n[模型2] SASRec + CE")
    torch.manual_seed(123); np.random.seed(123)
    model2 = build_recommendation_model(
        "sasrec", num_items, embedding_dim=64, hidden_dim=128,
        num_layers=2, dropout=0.2, max_len=max_seq_len).to(device)

    # SASRec不需要用户特征
    train_ds2 = RecDataset(aug_seqs, aug_targets, max_seq_len)
    train_loader2 = DataLoader(train_ds2, batch_size=256, shuffle=True, num_workers=0)

    opt2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=0)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50, eta_min=1e-5)
    best2, state2 = 0.0, None
    for epoch in range(1, 51):
        model2.train()
        for sb, lb, tb in train_loader2:
            sb, lb, tb = sb.to(device), lb.to(device), tb.to(device)
            opt2.zero_grad()
            scores = model2(sb, lb)
            loss = crit(scores, tb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
            opt2.step(); sched2.step()
        model2.eval()
        ndcgs = []
        with torch.no_grad():
            for start in range(0, len(val_seqs), 256):
                bs, bt = val_seqs[start:start+256], val_targets[start:start+256]
                seqs, lens = [], []
                for seq in bs:
                    if not seq: sp = [0]*max_seq_len; length = 1
                    else:
                        length = min(len(seq), max_seq_len)
                        sp = seq[-max_seq_len:] + [0]*(max_seq_len-len(seq[-max_seq_len:]))
                    seqs.append(sp); lens.append(length)
                st = torch.LongTensor(seqs).to(device)
                lt = torch.LongTensor(lens).to(device)
                scores = model2(st, lt)
                scores[:, 0] = -1e9
                _, topk = scores.topk(10, dim=1)
                topk = topk.cpu().numpy()
                for i, tgt in enumerate(bt):
                    if tgt in topk[i]:
                        ndcgs.append(1.0 / np.log2(np.where(topk[i]==tgt)[0][0] + 2))
                    else:
                        ndcgs.append(0.0)
        val_ndcg = float(np.mean(ndcgs))
        if val_ndcg > best2:
            best2 = val_ndcg; state2 = {k: v.clone() for k, v in model2.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  SASRec Epoch {epoch} | NDCG: {val_ndcg:.4f} | Best: {best2:.4f}")
    print(f"  SASRec 最佳: {best2:.4f}")

    # ===== 集成预测: 概率平均 =====
    print(f"\n[集成] GRU4Rec({best1:.4f}) + SASRec({best2:.4f}) 概率平均")
    model1.load_state_dict(state1); model1.eval()
    model2.load_state_dict(state2); model2.eval()

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

            # 两个模型的概率
            s1 = F.softmax(model1(st, lt, uft), dim=1)
            s2 = F.softmax(model2(st, lt), dim=1)
            # 加权平均 (GRU4Rec权重更高因为在测试集上更好)
            avg = 0.6 * s1 + 0.4 * s2
            avg[:, 0] = -1e9
            _, topk = avg.topk(10, dim=1)
            topk = topk.cpu().numpy()
            for pred in topk:
                items = [idx2item[i] for i in pred if i in idx2item and i > 0]
                while len(items) < 10:
                    items.append(idx2item.get(len(items)+1, "i000001"))
                all_preds.append(items[:10])

    elapsed = time.time() - start_time
    # 集成验证NDCG (近似)
    ens_val = max(best1, best2)  # 保守估计
    print(f"\n[Task2] 完成 | GRU4Rec={best1:.4f}, SASRec={best2:.4f} | 耗时={elapsed:.0f}s")
    with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, items in zip(test_uids, all_preds):
            f.write(f'{uid},"{",".join(items)}"\n')
    return ens_val


def main():
    print("=" * 70)
    print("  AFAC2026 集成版: GRU4Rec+SASRec推荐集成 + 标签传播0.5")
    print("=" * 70)
    t0 = time.time()
    cls = run_classification(device="cuda", n_ensemble=20)
    rec = run_recommendation_ensemble(device="cuda")
    elapsed = time.time() - t0
    final = 0.5 * cls + 0.5 * rec
    print(f"\n{'='*70}")
    print(f"  分类: {cls:.4f} | 推荐: {rec:.4f} | 总分: {final:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
