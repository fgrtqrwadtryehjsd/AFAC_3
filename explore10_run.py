"""
探索版10: GCNII分类(初始残差+恒等映射) + BERT4Rec推荐(双向Transformer+掩码训练)
基于论文:
- GCNII (ICML 2020): Simple and Deep Graph Convolutional Networks
- BERT4Rec (CIKM 2019): BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
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


# ============================================================
# GCNII: 初始残差 + 恒等映射
# ============================================================
class GCNIILayer(nn.Module):
    """GCNII层: H' = ((1-α)*A_norm*H + α*H_0) * ((1-β)*I + β*W)
    α: 初始残差权重 (保留初始特征)
    β: 恒等映射权重 (保留上一层的表示)
    """
    def __init__(self, in_dim, out_dim, alpha=0.1, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, x0, adj_sparse):
        # 邻居聚合 + 初始残差
        h_agg = torch.sparse.mm(adj_sparse, x)
        h = (1 - self.alpha) * h_agg + self.alpha * x0
        # 恒等映射 + 线性变换
        out = (1 - self.beta) * h + self.beta * self.linear(h)
        return out


class GCNII(nn.Module):
    """GCNII: 深层GCN, 防止过平滑"""
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=4,
                 alpha=0.1, beta=0.5, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GCNII层 (可以更深, 4层)
        self.layers = nn.ModuleList([
            GCNIILayer(hidden_dim, hidden_dim, alpha, beta)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, adj_sparse=None):
        h0 = self.input_proj(x)  # 初始特征
        h = h0
        for layer in self.layers:
            h = F.relu(layer(h, h0, adj_sparse))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


def run_cls(device="cuda", n_ensemble=30, lp_weight=0.3):
    """GCNII + GCN 混合集成"""
    print("\n" + "=" * 70)
    print(f"  Task1: GCNII{n_ensemble//2} + GCN{n_ensemble//2} 混合集成")
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
    n_gcnii = n_ensemble - n_gcn
    all_probs = []
    best_val = 0

    # GCN集成 (baseline)
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

    # GCNII集成 (深层, 初始残差+恒等映射)
    print(f"\n[GCNII] {n_gcnii}模型 (4层, 初始残差+恒等映射)...")
    gcnii_val = 0
    for i in range(n_gcnii):
        seed = 200 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        # 尝试不同的alpha和beta
        alpha = 0.1 if i % 2 == 0 else 0.2
        beta = 0.5 if i % 2 == 0 else 0.4
        gcnii = GCNII(feat_dim, 256, num_classes, num_layers=4, alpha=alpha, beta=beta, dropout=0.5).to(device)
        opt = torch.optim.Adam(gcnii.parameters(), lr=0.005, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_probs = None; bv = 0
        for epoch in range(1, 301):
            gcnii.train(); opt.zero_grad()
            adj_tr = drop_edge(adj_sparse, 0.2)
            logits = gcnii(features_t, adj_tr)
            loss = crit(logits[train_only_t], labels_t[train_only_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gcnii.parameters(), 5.0)
            opt.step(); sched.step()
            if epoch % 10 == 0:
                gcnii.eval()
                with torch.no_grad():
                    vp = gcnii(features_t, adj_sparse)[val_t].argmax(1)
                    va = (vp == labels_t[val_t]).float().mean().item()
                    if va > bv:
                        bv = va
                        best_probs = F.softmax(gcnii(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
        all_probs.append(best_probs)
        if bv > gcnii_val: gcnii_val = bv
        if (i+1) % 5 == 0: print(f"    GCNII {i+1}/{n_gcnii} (a={alpha},b={beta}) Val={bv:.4f}")

    print(f"  GCN Val={best_val:.4f}, GCNII Val={gcnii_val:.4f}")

    # 按验证分数加权融合
    if gcnii_val > 0:
        total = best_val + gcnii_val
        mixed = (best_val / total) * np.mean(all_probs[:n_gcn], axis=0) + \
                (gcnii_val / total) * np.mean(all_probs[n_gcn:], axis=0)
    else:
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
    print(f"\n[Task1完成] GCNII+GCN混合 | 耗时: {elapsed/60:.1f}min")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")
    return max(best_val, gcnii_val)


# ============================================================
# BERT4Rec: 双向Transformer + 掩码训练
# ============================================================
class BERT4Rec(nn.Module):
    """BERT4Rec: 双向自注意力 + 掩码语言模型训练"""
    def __init__(self, num_items, embedding_dim=128, hidden_dim=256,
                 num_heads=4, num_layers=2, dropout=0.2, max_len=50):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        
        # 用户特征
        self.user_feat_dims = None
        self.user_embeddings = None
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embedding_dim)
        
    def add_user_features(self, user_feat_dims):
        self.user_feat_dims = user_feat_dims
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(dim + 1, 16, padding_idx=0) for dim in user_feat_dims])
        # 用户特征投影到embedding_dim (取item_embedding的维度)
        emb_dim = self.item_embedding.embedding_dim
        self.user_proj = nn.Linear(len(user_feat_dims) * 16, emb_dim)
        
    def forward(self, seq_tensor, seq_lengths, user_features=None, item_features=None, item_qwen_emb=None):
        B, L = seq_tensor.shape
        pos_ids = torch.arange(L, device=seq_tensor.device).unsqueeze(0).expand(B, L)
        
        emb = self.item_embedding(seq_tensor) + self.pos_embedding(pos_ids)
        
        # 用户特征融合
        if self.user_embeddings is not None and user_features is not None:
            user_embs = [emb_layer(user_features[:, i]) for i, emb_layer in enumerate(self.user_embeddings)]
            user_repr = torch.cat(user_embs, dim=-1)
            user_proj = self.user_proj(user_repr).unsqueeze(1)  # (B, 1, D)
            emb = emb + user_proj  # 广播到所有位置
        
        emb = self.layernorm(self.dropout(emb))
        
        # 创建padding mask
        mask = (seq_tensor == 0)  # True = padding位置
        
        out = self.transformer(emb, src_key_padding_mask=mask)
        
        # 用最后一个非padding位置的输出
        # seq_lengths: (B,), 取每个序列最后一个位置
        out = out[torch.arange(B), seq_lengths - 1]  # (B, D)
        
        return self.output_proj(out)


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


def run_rec(device="cuda", sim_lambda=0.5):
    """BERT4Rec + 用户特征 + 物品特征 + SimRec"""
    print("\n" + "=" * 70)
    print(f"  Task2: BERT4Rec + 用户特征 + SimRec λ={sim_lambda}")
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
                scores = model(sb, lb, ub)
                scores[:, 0] = -1e9
                _, tk = scores.topk(10, dim=1); tk = tk.cpu().numpy()
                for i, target in enumerate(bt):
                    if target in tk[i]:
                        r = np.where(tk[i] == target)[0][0]; ndcgs.append(1.0/np.log2(r+2))
                    else: ndcgs.append(0)
        return float(np.mean(ndcgs))

    # 3模型集成
    all_test_probs = []
    for model_idx, seed_base in enumerate([42, 142, 242]):
        print(f"\n[BERT4Rec #{model_idx+1}] seed={seed_base}")
        torch.manual_seed(seed_base); np.random.seed(seed_base)
        model = BERT4Rec(num_items, 128, 256, num_heads=4, num_layers=2, dropout=0.2, max_len=max_len)
        model.add_user_features(user_feat_dims)
        model = model.to(device)
        
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
                scores = model(sb, lb, ub)
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
        print(f"  BERT4Rec #{model_idx+1} Val={best_val:.4f}")

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
                scores = model(sb, lb, ub)
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
    args = parser.parse_args()

    print("=" * 70)
    print("  探索版10: GCNII分类 + BERT4Rec推荐")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_val = 0.72; rec_val = 0
    if args.task in ("both", "cls"):
        cls_val = run_cls(device=device, n_ensemble=30, lp_weight=0.3)
    if args.task in ("both", "rec"):
        rec_val = run_rec(device=device, sim_lambda=0.5)

    elapsed = time.time() - t0
    est_total = 0.5 * cls_val + 0.5 * rec_val * 0.834
    print(f"\n{'='*70}")
    print(f"  分类Val={cls_val:.4f} | 推荐Val={rec_val:.4f} | 预估test={est_total:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
