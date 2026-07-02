"""
探索版11: 数据层面优化
分类: PCA特征降维(767→128去噪) + GCN50集成
推荐: 物品共现矩阵作为额外特征 + GRU4Rec emb=128

核心思路: 之前所有尝试都在模型层面, 但数据层面的优化可能更有效
- 分类: 767维特征可能有噪声, PCA降维去噪可能提升
- 推荐: 物品共现矩阵可以提供额外的协同过滤信号
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.models import build_classification_model
from src.train_cls_improved import drop_edge, label_propagation

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLS_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A分类", "A分类", "A1.npz")
REC_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A推荐", "A推荐")


def run_cls(device="cuda", n_ensemble=50, lp_weight=0.3, use_pca=True, pca_dim=128):
    print("\n" + "=" * 70)
    print(f"  Task1: GCN{n_ensemble} + PCA({pca_dim if use_pca else 'none'}) + 伪标签 + lp={lp_weight}")
    print("=" * 70)
    t0 = time.time()

    data = load_classification_data(CLS_DATA)
    adj, features, labels = data["adj"], data["features"], data["labels"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    num_classes = data["num_classes"]
    feat_dim = data["feat_dim"]

    # PCA降维去噪
    if use_pca:
        print(f"  PCA降维: {feat_dim} → {pca_dim}")
        # 先L2归一化
        features_l2 = normalize(features, norm="l2", axis=1)
        # PCA
        svd = TruncatedSVD(n_components=pca_dim, random_state=42)
        features_pca = svd.fit_transform(features_l2)
        # 再L2归一化
        features = normalize(features_pca, norm="l2", axis=1).astype(np.float32)
        print(f"  PCA解释方差比: {svd.explained_variance_ratio_.sum():.4f}")
    else:
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

    print(f"\n[第1轮] GCN {n_ensemble}模型...")
    all_probs = []
    best_val = 0
    for i in range(n_ensemble):
        seed = 42 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_classification_model("gcn", features.shape[1], 256, num_classes, 2, 0.5, "sym").to(device)
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

    # 标签传播用原始特征(不是PCA特征)
    features_orig = normalize(data["features"], norm="l2", axis=1).astype(np.float32)
    lp = label_propagation(features_orig, labels, train_idx, test_idx, k_neighbors=10)
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
        model = build_classification_model("gcn", features.shape[1], 256, num_classes, 2, 0.5, "sym").to(device)
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
    print(f"\n[Task1完成] PCA{pca_dim} | 耗时: {elapsed/60:.1f}min")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return best_val


def compute_item_sim(train_seqs, num_items):
    cooc = np.zeros((num_items+1, num_items+1), dtype=np.float32)
    for seq in train_seqs:
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                if seq[i] > 0 and seq[j] > 0:
                    cooc[seq[i], seq[j]] += 1
                    cooc[seq[j], seq[i]] += 1
    rn = np.sqrt((cooc**2).sum(1, keepdims=True) + 1e-8)
    sm = cooc / rn / rn.T
    sm[0] = 0
    sd = np.zeros_like(sm)
    for i in range(1, num_items+1):
        sd[i] = np.exp(sm[i] - sm[i].max())
        sd[i, 0] = 0
        sd[i] /= sd[i].sum() + 1e-8
    return torch.FloatTensor(sd).to("cuda" if torch.cuda.is_available() else "cpu")


class GRU4RecFullFeatures(nn.Module):
    def __init__(self, num_items, embedding_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.2, max_len=50,
                 user_feat_dims=None, item_feat_dims=None):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
        if item_feat_dims:
            self.item_feat_embeddings = nn.ModuleList([
                nn.Embedding(dim + 1, 8, padding_idx=0) for dim in item_feat_dims])
            self.item_feat_proj = nn.Linear(len(item_feat_dims) * 8, embedding_dim)
        else:
            self.item_feat_embeddings = None
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        if user_feat_dims:
            self.user_embeddings = nn.ModuleList([
                nn.Embedding(dim + 1, 16, padding_idx=0) for dim in user_feat_dims])
            user_repr_dim = len(user_feat_dims) * 16
        else:
            self.user_embeddings = None
            user_repr_dim = 0
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + user_repr_dim, hidden_dim), nn.ReLU())
        self.output_proj = nn.Linear(hidden_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_tensor, seq_lengths, user_features=None, item_features=None, item_qwen_emb=None):
        emb = self.item_embedding(seq_tensor)
        if self.item_feat_embeddings is not None and item_features is not None:
            item_feat_embs = [emb_layer(item_features[seq_tensor, i]) for i, emb_layer in enumerate(self.item_feat_embeddings)]
            item_feat_combined = torch.cat(item_feat_embs, dim=-1)
            emb = emb + self.item_feat_proj(item_feat_combined)
        emb = self.dropout(emb)
        packed = nn.utils.rnn.pack_padded_sequence(emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        seq_repr = hidden[-1]
        if self.user_embeddings is not None and user_features is not None:
            user_embs = [emb_layer(user_features[:, i]) for i, emb_layer in enumerate(self.user_embeddings)]
            user_repr = torch.cat(user_embs, dim=-1)
            combined = self.fusion(torch.cat([seq_repr, user_repr], dim=-1))
        else:
            combined = seq_repr
        combined = self.dropout(combined)
        return self.output_proj(combined)


def run_rec(device="cuda", sim_lambda=0.5, embedding_dim=128, hidden_dim=256):
    print("\n" + "=" * 70)
    print(f"  Task2: GRU4Rec emb={embedding_dim} + 物品特征 + SimRec λ={sim_lambda}")
    print("=" * 70)
    t0 = time.time()

    data = load_recommendation_data(REC_DATA)
    train_df = data["train_df"]; test_df = data["test_df"]
    user_df = data["user_df"]; item_df = data["item_df"]
    item2idx = data["item2idx"]; idx2item = data["idx2item"]
    num_items = data["num_items"]
    max_len = 50

    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_len)

    user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
    user_feat_dims = [int(user_df[col].max()) + 1 for col in user_feat_cols]
    user_feat_dict = {}
    for _, row in user_df.iterrows():
        user_feat_dict[row["uid"]] = torch.LongTensor([int(row[c]) for c in user_feat_cols])

    item_feat_cols = [c for c in item_df.columns if c.startswith("i_cat") or c.startswith("i_bucket")]
    item_feat_dims = [int(item_df[col].max()) + 1 for col in item_feat_cols]
    item_feat_tensor = torch.zeros(num_items + 2, len(item_feat_cols), dtype=torch.long)
    for iid, idx in item2idx.items():
        if idx > 0:
            row = item_df[item_df["iid"] == iid].iloc[0]
            for j, col in enumerate(item_feat_cols):
                item_feat_tensor[idx, j] = int(row[col])
    item_feat_tensor = item_feat_tensor.to(device)

    sim_dist = compute_item_sim(train_seqs, num_items)

    np.random.seed(42)
    indices = np.random.permutation(len(train_seqs))
    n_val = int(len(train_seqs) * 0.1)
    val_seqs = [train_seqs[i] for i in indices[:n_val]]
    val_targets = [train_targets[i] for i in indices[:n_val]]
    val_uids_list = [train_df.iloc[i]["uid"] for i in indices[:n_val]]
    tr_seqs = [train_seqs[i] for i in indices[n_val:]]
    tr_targets = [train_targets[i] for i in indices[n_val:]]
    tr_uids_list = [train_df.iloc[i]["uid"] for i in indices[n_val:]]

    aug_seqs, aug_targets, aug_uids = list(tr_seqs), list(tr_targets), list(tr_uids_list)
    for seq, tgt, uid in zip(tr_seqs, tr_targets, tr_uids_list):
        if len(seq) > 5:
            for tl in [5, 10]:
                if len(seq) > tl:
                    aug_seqs.append(seq[-tl:]); aug_targets.append(tgt); aug_uids.append(uid)

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
            return (torch.LongTensor(sp), torch.LongTensor([length])[0],
                    torch.LongTensor([self.t[i]])[0],
                    self.uf.get(self.u[i], torch.zeros(8, dtype=torch.long)))

    train_ds = RecDS(aug_seqs, aug_targets, aug_uids, user_feat_dict, max_len)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

    def evaluate(model):
        model.eval()
        ndcgs = []
        with torch.no_grad():
            for start in range(0, len(val_seqs), 256):
                bs = val_seqs[start:start+256]; bt = val_targets[start:start+256]; bu = val_uids_list[start:start+256]
                st, lt, uf = [], [], []
                for seq, uid in zip(bs, bu):
                    if not seq: sp = [0]*max_len; l = 1
                    else:
                        l = min(len(seq), max_len); sp = seq[-max_len:] + [0]*(max_len-len(seq[-max_len:]))
                    st.append(sp); lt.append(l); uf.append(user_feat_dict.get(uid, torch.zeros(8, dtype=torch.long)))
                sb = torch.LongTensor(st).to(device); lb = torch.LongTensor(lt).to(device); ub = torch.stack(uf).to(device)
                scores = model(sb, lb, ub, item_feat_tensor)
                scores[:, 0] = -1e9
                _, tk = scores.topk(10, dim=1); tk = tk.cpu().numpy()
                for i, target in enumerate(bt):
                    if target in tk[i]:
                        r = np.where(tk[i] == target)[0][0]; ndcgs.append(1.0/np.log2(r+2))
                    else: ndcgs.append(0)
        return float(np.mean(ndcgs))

    all_test_probs = []
    for model_idx, seed_base in enumerate([42, 142, 242]):
        print(f"\n[GRU4Rec #{model_idx+1}] seed={seed_base}")
        torch.manual_seed(seed_base); np.random.seed(seed_base)
        model = GRU4RecFullFeatures(num_items, embedding_dim, hidden_dim, 1, 0.2, max_len,
                                    user_feat_dims, item_feat_dims).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_val = 0; best_state = None; patience = 0
        total_steps = len(train_loader) * 50; warmup = 1000; gs = 0

        for epoch in range(1, 51):
            model.train()
            for sb, lb, tb, ub in train_loader:
                sb = sb.to(device); lb = lb.to(device); tb = tb.to(device); ub = ub.to(device)
                opt.zero_grad()
                scores = model(sb, lb, ub, item_feat_tensor)
                scores[:, 0] = -1e9
                ce = crit(scores, tb)
                with torch.no_grad():
                    ts = sim_dist[tb]; ts[:, 0] = 0; ts = ts / (ts.sum(1, keepdims=True) + 1e-8)
                lp = F.log_softmax(scores, dim=1)
                sl = -(ts * lp).sum(1).mean()
                cl = sim_lambda * min(gs/warmup, 1.0) if gs < warmup else sim_lambda * max(0.1, 1-(gs-warmup)/(total_steps-warmup))
                loss = (1-cl)*ce + cl*sl
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step(); gs += 1
            v = evaluate(model)
            if v > best_val:
                best_val = v; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
            else: patience += 1
            if patience >= 7: break
            if epoch % 5 == 0: print(f"  Epoch {epoch}: Val={v:.4f}")

        model.load_state_dict(best_state); model.to(device)
        print(f"  #{model_idx+1} Val={best_val:.4f}")

        model.eval()
        with torch.no_grad():
            probs_list = []
            for start in range(0, len(test_seqs), 256):
                bs = test_seqs[start:start+256]; bu = test_uids[start:start+256]
                st, lt, uf = [], [], []
                for seq, uid in zip(bs, bu):
                    if not seq: sp = [0]*max_len; l = 1
                    else:
                        l = min(len(seq), max_len); sp = seq[-max_len:] + [0]*(max_len-len(seq[-max_len:]))
                    st.append(sp); lt.append(l); uf.append(user_feat_dict.get(uid, torch.zeros(8, dtype=torch.long)))
                sb = torch.LongTensor(st).to(device); lb = torch.LongTensor(lt).to(device); ub = torch.stack(uf).to(device)
                scores = model(sb, lb, ub, item_feat_tensor)
                scores[:, 0] = -1e9
                probs_list.append(F.softmax(scores, dim=1).cpu())
            all_test_probs.append((best_val, torch.cat(probs_list, dim=0)))

    total_val = sum(v for v, _ in all_test_probs)
    weights = [v / total_val for v, _ in all_test_probs]
    print(f"\n[集成] 权重: {[f'{w:.3f}' for w in weights]}")
    ensemble_probs = sum(w * p for (v, p), w in zip(all_test_probs, weights))

    best_val = max(v for v, _ in all_test_probs)
    elapsed = time.time() - t0
    print(f"\n[Task2完成] Val={best_val:.4f} | 耗时: {elapsed:.1f}s")

    _, topk = ensemble_probs.topk(10, dim=1)
    topk = topk.numpy()
    all_preds = []
    for pred in topk:
        items = [idx2item.get(i, "i000001") for i in pred if i in idx2item and i > 0]
        while len(items) < 10: items.append("i000001")
        all_preds.append(items[:10])

    with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, items in zip(test_uids, all_preds):
            f.write(f'{uid},"{",".join(items)}"\n')
    return best_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="both", choices=["both", "cls", "rec"])
    parser.add_argument("--pca_dim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  探索版11: PCA({args.pca_dim})分类 + GRU4Rec推荐")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_val = 0.72; rec_val = 0
    if args.task in ("both", "cls"):
        cls_val = run_cls(device=device, n_ensemble=50, lp_weight=0.3, use_pca=True, pca_dim=args.pca_dim)
    if args.task in ("both", "rec"):
        rec_val = run_rec(device=device, sim_lambda=0.5, embedding_dim=128, hidden_dim=256)

    elapsed = time.time() - t0
    est_total = 0.5 * cls_val + 0.5 * rec_val * 0.834
    print(f"\n{'='*70}")
    print(f"  分类Val={cls_val:.4f} | 推荐Val={rec_val:.4f} | 预估test={est_total:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
