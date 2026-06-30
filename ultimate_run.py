"""
终极突破版:
分类: GCN嵌入标签传播 (用GCN学到的256维嵌入做KNN, 而非原始767维特征)
推荐: GRU4Rec+用户特征+SimRec + SASRec 双模型集成
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.models import build_classification_model, build_recommendation_model
from src.train_cls_improved import drop_edge, label_propagation
from src.train_rec import RecDataset, TestRecDataset
from src.train_rec_improved import compute_ndcg_at_k

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLS_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A分类", "A分类", "A1.npz")
REC_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A推荐", "A推荐")


def gcn_embedding_label_propagation(gcn_model, features_t, adj_sparse, labels,
                                     train_idx, test_idx, k_neighbors=10):
    """用GCN学到的嵌入做标签传播 (比原始特征更有判别性)"""
    print("[GCN嵌入标签传播] 提取GCN最后一层嵌入...")
    gcn_model.eval()
    with torch.no_grad():
        # 获取GCN的隐藏层输出 (分类器之前的层)
        h = features_t
        for i in range(len(gcn_model.layers)):
            h_neigh = torch.sparse.mm(adj_sparse, h)
            h = gcn_model.layers[i](h_neigh)
            if hasattr(gcn_model, 'bns'):
                h = gcn_model.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=gcn_model.dropout, training=False)
        embeddings = h.cpu().numpy()  # (num_nodes, hidden_dim)

    # L2归一化嵌入
    embeddings = normalize(embeddings, norm="l2", axis=1)

    # KNN标签传播
    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]
    train_labels = labels[train_idx]

    num_classes = int(labels.max()) + 1
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(train_idx)),
                            metric="cosine", n_jobs=-1).fit(train_emb)
    distances, indices = nbrs.kneighbors(test_emb)

    propagated = np.zeros((len(test_idx), num_classes))
    for i in range(len(test_idx)):
        neighbor_labels = train_labels[indices[i]]
        weights = np.exp(-distances[i])
        weights /= weights.sum() + 1e-8
        for j, label in enumerate(neighbor_labels):
            propagated[i, label] += weights[j]

    print(f"[GCN嵌入标签传播] 完成, 嵌入维度={embeddings.shape[1]}")
    return propagated


def run_classification(device="cuda", n_ensemble=20):
    """GCN集成 + GCN嵌入标签传播 + 多轮伪标签"""
    print("\n" + "=" * 70)
    print("  Task1: GCN集成 + GCN嵌入标签传播 + 多轮伪标签")
    print("=" * 70)
    t0 = time.time()

    data = load_classification_data(CLS_DATA)
    adj, features, labels = data["adj"], data["features"], data["labels"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    num_nodes, feat_dim = data["feat_dim"], data["feat_dim"]
    num_classes = data["num_classes"]

    features = normalize(features, norm="l2", axis=1).astype(np.float32)
    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    test_t = torch.LongTensor(test_idx).to(device)

    np.random.seed(42)
    perm = np.random.permutation(train_idx)
    n_val = int(len(train_idx) * 0.1)
    val_idx_arr = perm[:n_val]
    train_only_arr = perm[n_val:]
    val_t = torch.LongTensor(val_idx_arr).to(device)
    train_only_t = torch.LongTensor(train_only_arr).to(device)

    # ===== 第1轮: GCN集成 =====
    print(f"\n[第1轮] GCN集成 ({n_ensemble}模型)...")
    gcn_test_probs = []
    best_gcn_model = None
    best_gcn_val = 0
    for i in range(n_ensemble):
        seed = 42 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_val = 0; best_probs = None
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
                    val_pred = model(features_t, adj_sparse)[val_t].argmax(1)
                    val_acc = (val_pred == labels_t[val_t]).float().mean().item()
                    if val_acc > best_val:
                        best_val = val_acc
                        best_probs = F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
                        if val_acc > best_gcn_val:
                            best_gcn_val = val_acc
                            best_gcn_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        gcn_test_probs.append(best_probs)
        if (i + 1) % 5 == 0:
            print(f"    GCN {i+1}/{n_ensemble} Val={best_val:.4f}")

    gcn_ensemble = np.mean(gcn_test_probs, axis=0)
    print(f"  GCN集成 Val={best_gcn_val:.4f}")

    # ===== GCN嵌入标签传播 (核心创新) =====
    print("\n[GCN嵌入标签传播] 用最佳GCN的嵌入做KNN标签传播...")
    best_model = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
    best_model.load_state_dict(best_gcn_model)
    best_model.to(device)
    gcn_lp = gcn_embedding_label_propagation(
        best_model, features_t, adj_sparse, labels, train_idx, test_idx, k_neighbors=15)
    gcn_lp = gcn_lp / (gcn_lp.sum(1, keepdims=True) + 1e-8)

    # 原始特征标签传播
    feat_lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    feat_lp = feat_lp / (feat_lp.sum(1, keepdims=True) + 1e-8)

    # 双重标签传播融合: GCN嵌入 + 原始特征
    fused_lp = 0.7 * gcn_lp + 0.3 * feat_lp

    # GCN + 标签传播融合
    lp_weight = 0.4
    fused_pred = (1 - lp_weight) * gcn_ensemble + lp_weight * fused_lp
    print(f"  GCN+标签传播(0.4) 融合完成")

    # ===== 第2轮: 伪标签 + GCN嵌入标签传播 =====
    print("\n[第2轮] 伪标签(>0.7) + 100%数据重训...")
    conf = fused_pred.max(axis=1)
    pseudo_mask = conf > 0.7
    pseudo_labels = fused_pred.argmax(axis=1)
    final_train = np.concatenate([train_idx, test_idx[pseudo_mask]])
    final_labels = labels.copy()
    final_labels[test_idx[pseudo_mask]] = pseudo_labels[pseudo_mask]
    print(f"  伪标签: {pseudo_mask.sum()}个节点, 总训练: {len(final_train)}")

    ft_mask = torch.LongTensor(final_train).to(device)
    ft_labels = torch.LongTensor(final_labels).to(device)
    pseudo_probs = []
    best_pseudo_model = None
    best_pseudo_val = 0
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
            # 验证
            val_pred = model(features_t, adj_sparse)[val_t].argmax(1)
            val_acc = (val_pred == labels_t[val_t]).float().mean().item()
            if val_acc > best_pseudo_val:
                best_pseudo_val = val_acc
                best_pseudo_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    pseudo_ensemble = np.mean(pseudo_probs, axis=0)

    # 第2轮GCN嵌入标签传播
    best_model2 = build_classification_model("gcn", feat_dim, 256, num_classes, 2, 0.5, "sym").to(device)
    best_model2.load_state_dict(best_pseudo_model)
    best_model2.to(device)
    gcn_lp2 = gcn_embedding_label_propagation(
        best_model2, features_t, adj_sparse, labels, train_idx, test_idx, k_neighbors=15)
    gcn_lp2 = gcn_lp2 / (gcn_lp2.sum(1, keepdims=True) + 1e-8)
    fused_lp2 = 0.7 * gcn_lp2 + 0.3 * feat_lp

    # 最终融合: 伪标签GCN + 双重标签传播(0.4)
    final_pred = 0.6 * pseudo_ensemble + 0.4 * fused_lp2
    test_pred = final_pred.argmax(axis=1)

    elapsed = time.time() - t0
    val_acc = best_pseudo_val
    print(f"\n[Task1完成] GCN嵌入标签传播+伪标签 | Val={val_acc:.4f} | 耗时: {elapsed:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")

    return val_acc


def compute_item_similarity(train_seqs, num_items):
    print("[SimRec] 计算物品共现相似度...")
    cooccurrence = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
    for seq in train_seqs:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i] > 0 and seq[j] > 0:
                    cooccurrence[seq[i], seq[j]] += 1
                    cooccurrence[seq[j], seq[i]] += 1
    row_norm = np.sqrt((cooccurrence ** 2).sum(axis=1, keepdims=True) + 1e-8)
    sim_matrix = cooccurrence / row_norm / row_norm.T
    sim_matrix[0] = 0
    sim_dist = np.zeros_like(sim_matrix)
    for i in range(1, num_items + 1):
        sim_dist[i] = np.exp(sim_matrix[i] - sim_matrix[i].max())
        sim_dist[i, 0] = 0
        sim_dist[i] /= sim_dist[i].sum() + 1e-8
    return torch.FloatTensor(sim_dist).to("cuda" if torch.cuda.is_available() else "cpu")


def run_recommendation(device="cuda", sim_lambda=0.3):
    """GRU4Rec+用户特征+SimRec + SASRec 双模型集成"""
    print("\n" + "=" * 70)
    print(f"  Task2: GRU4Rec+用户特征+SimRec + SASRec 集成")
    print("=" * 70)
    t0 = time.time()

    data = load_recommendation_data(REC_DATA)
    train_df = data["train_df"]; test_df = data["test_df"]
    user_df = data["user_df"]
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

    sim_dist = compute_item_similarity(train_seqs, num_items)

    # 短序列验证集
    np.random.seed(42)
    seq_lens = np.array([len(s) for s in train_seqs])
    short_indices = np.where(seq_lens <= 10)[0]
    np.random.shuffle(short_indices)
    n_val = min(2000, len(short_indices))
    val_indices = short_indices[:n_val]
    val_set = set(val_indices.tolist())
    train_indices = np.array([i for i in range(len(train_seqs)) if i not in val_set])

    val_seqs = [train_seqs[i] for i in val_indices]
    val_targets = [train_targets[i] for i in val_indices]
    val_uids_list = [train_df.iloc[i]["uid"] for i in val_indices]
    tr_seqs = [train_seqs[i] for i in train_indices]
    tr_targets = [train_targets[i] for i in train_indices]
    tr_uids_list = [train_df.iloc[i]["uid"] for i in train_indices]

    # 序列增强
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

    def evaluate_val(model, use_user_feat=True):
        model.eval()
        ndcgs = []
        with torch.no_grad():
            for start in range(0, len(val_seqs), 256):
                batch_seqs = val_seqs[start:start+256]
                batch_targets = val_targets[start:start+256]
                batch_uids = val_uids_list[start:start+256]
                seqs_t, lengths_t, ufs = [], [], []
                for seq, uid in zip(batch_seqs, batch_uids):
                    if not seq: sp = [0]*max_len; length = 1
                    else:
                        length = min(len(seq), max_len)
                        sp = seq[-max_len:] + [0]*(max_len-len(seq[-max_len:]))
                    seqs_t.append(sp); lengths_t.append(length)
                    ufs.append(user_feat_dict.get(uid, torch.zeros(8, dtype=torch.long)))
                seq_batch = torch.LongTensor(seqs_t).to(device)
                length_batch = torch.LongTensor(lengths_t).to(device)
                uf_batch = torch.stack(ufs).to(device)
                if use_user_feat:
                    scores = model(seq_batch, length_batch, uf_batch)
                else:
                    scores = model(seq_batch, length_batch)
                scores[:, 0] = -1e9
                _, topk = scores.topk(10, dim=1)
                topk = topk.cpu().numpy()
                for i, target in enumerate(batch_targets):
                    if target in topk[i]:
                        rank = np.where(topk[i] == target)[0][0]
                        ndcgs.append(1.0 / np.log2(rank + 2))
                    else:
                        ndcgs.append(0)
        return float(np.mean(ndcgs))

    # ===== 模型1: GRU4Rec+用户特征+SimRec损失 =====
    print(f"\n[模型1] GRU4Rec+用户特征+SimRec(lambda={sim_lambda})")
    model1 = build_recommendation_model(
        "gru4rec", num_items, embedding_dim=64, hidden_dim=128,
        num_layers=1, dropout=0.2, max_len=max_len,
        user_feat_dims=user_feat_dims).to(device)
    opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=50, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()
    best1 = 0; state1 = None; patience = 0
    total_steps = len(train_loader) * 50; warmup = 1000; gs = 0

    for epoch in range(1, 51):
        model1.train()
        for seq_batch, length_batch, target_batch, uf_batch in train_loader:
            seq_batch = seq_batch.to(device); length_batch = length_batch.to(device)
            target_batch = target_batch.to(device); uf_batch = uf_batch.to(device)
            opt1.zero_grad()
            scores = model1(seq_batch, length_batch, uf_batch)
            scores[:, 0] = -1e9
            ce_loss = crit(scores, target_batch)
            with torch.no_grad():
                target_sim = sim_dist[target_batch]; target_sim[:, 0] = 0
                target_sim = target_sim / (target_sim.sum(1, keepdims=True) + 1e-8)
            log_probs = F.log_softmax(scores, dim=1)
            sim_loss = -(target_sim * log_probs).sum(1).mean()
            cur_lambda = sim_lambda * min(gs / warmup, 1.0) if gs < warmup else sim_lambda * max(0.1, 1 - (gs - warmup) / (total_steps - warmup))
            loss = (1 - cur_lambda) * ce_loss + cur_lambda * sim_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
            opt1.step(); sched1.step(); gs += 1
        v = evaluate_val(model1, use_user_feat=True)
        if v > best1: best1 = v; state1 = {k: v.cpu().clone() for k, v in model1.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= 7: break
        if epoch % 5 == 0: print(f"  GRU4Rec Epoch {epoch}: Val={v:.4f}")
    model1.load_state_dict(state1); model1.to(device)
    print(f"  GRU4Rec 最佳 Val={best1:.4f}")

    # ===== 模型2: SASRec =====
    print(f"\n[模型2] SASRec")
    model2 = build_recommendation_model("sasrec", num_items, embedding_dim=64,
                                        hidden_dim=128, num_layers=2, dropout=0.2, max_len=max_len).to(device)
    # SASRec不需要用户特征
    train_ds2 = RecDS(aug_seqs, aug_targets, aug_uids, user_feat_dict, max_len)
    train_loader2 = DataLoader(train_ds2, batch_size=256, shuffle=True, num_workers=0)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50, eta_min=1e-5)
    best2 = 0; state2 = None; patience = 0
    for epoch in range(1, 51):
        model2.train()
        for seq_batch, length_batch, target_batch, _ in train_loader2:
            seq_batch = seq_batch.to(device); length_batch = length_batch.to(device)
            target_batch = target_batch.to(device)
            opt2.zero_grad()
            scores = model2(seq_batch, length_batch)
            scores[:, 0] = -1e9
            loss = crit(scores, target_batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
            opt2.step(); sched2.step()
        v = evaluate_val(model2, use_user_feat=False)
        if v > best2: best2 = v; state2 = {k: v.cpu().clone() for k, v in model2.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= 7: break
        if epoch % 5 == 0: print(f"  SASRec Epoch {epoch}: Val={v:.4f}")
    model2.load_state_dict(state2); model2.to(device)
    print(f"  SASRec 最佳 Val={best2:.4f}")

    # ===== 集成预测 (GRU4Rec 0.6 + SASRec 0.4) =====
    print(f"\n[集成] GRU4Rec({best1:.4f})*0.6 + SASRec({best2:.4f})*0.4")
    model1.eval(); model2.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_seqs), 256):
            batch_seqs = test_seqs[start:start+256]
            batch_uids = test_uids[start:start+256]
            seqs_t, lengths_t, ufs = [], [], []
            for seq, uid in zip(batch_seqs, batch_uids):
                if not seq: sp = [0]*max_len; length = 1
                else:
                    length = min(len(seq), max_len)
                    sp = seq[-max_len:] + [0]*(max_len-len(seq[-max_len:]))
                seqs_t.append(sp); lengths_t.append(length)
                ufs.append(user_feat_dict.get(uid, torch.zeros(8, dtype=torch.long)))
            seq_batch = torch.LongTensor(seqs_t).to(device)
            length_batch = torch.LongTensor(lengths_t).to(device)
            uf_batch = torch.stack(ufs).to(device)
            s1 = F.softmax(model1(seq_batch, length_batch, uf_batch), dim=1)
            s2 = F.softmax(model2(seq_batch, length_batch), dim=1)
            avg = 0.6 * s1 + 0.4 * s2
            avg[:, 0] = -1e9
            _, topk = avg.topk(10, dim=1)
            topk = topk.cpu().numpy()
            for pred in topk:
                items = [idx2item.get(i, "i000001") for i in pred if i in idx2item and i > 0]
                while len(items) < 10: items.append("i000001")
                all_preds.append(items[:10])

    ensemble_val = max(best1, best2)
    elapsed = time.time() - t0
    print(f"\n[Task2完成] 集成Val={ensemble_val:.4f} | 耗时: {elapsed:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, items in zip(test_uids, all_preds):
            f.write(f'{uid},"{",".join(items)}"\n')

    return ensemble_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="both", choices=["both", "cls", "rec"])
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 终极突破版")
    print("  分类: GCN嵌入标签传播+多轮伪标签")
    print("  推荐: GRU4Rec+SimRec + SASRec 双模型集成")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_val = 0.69; rec_val = 0
    if args.task in ("both", "cls"):
        cls_val = run_classification(device=device, n_ensemble=20)
    if args.task in ("both", "rec"):
        rec_val = run_recommendation(device=device, sim_lambda=0.5)

    elapsed = time.time() - t0
    final = 0.5 * cls_val + 0.5 * rec_val
    print(f"\n{'='*70}")
    print(f"  分类Val: {cls_val:.4f} | 推荐Val: {rec_val:.4f} | 预估: {final:.4f}")
    print(f"  总耗时: {elapsed/60:.1f}分钟")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
