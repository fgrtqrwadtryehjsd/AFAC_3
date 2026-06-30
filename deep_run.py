"""
深度突破版:
分类: 度数自适应融合 - 孤立节点(度<=1)用MLP, 有邻居节点用GCN, 按度数加权
推荐: SimRec损失 + 用户特征融合 + 短序列验证集(匹配测试分布)
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.models import build_classification_model, build_recommendation_model, MLPCls
from src.train_cls_improved import drop_edge, label_propagation
from src.train_rec import RecDataset, TestRecDataset
from src.train_rec_improved import compute_ndcg_at_k

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLS_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A分类", "A分类", "A1.npz")
REC_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A推荐", "A推荐")


# ============================================================
# 分类: 度数自适应融合
# ============================================================
def run_classification_adaptive(device="cuda", n_gcn=15, n_mlp=10):
    """度数自适应融合: 不是简单集成, 而是按节点度数加权GCN和MLP的预测"""
    print("\n" + "=" * 70)
    print("  Task1: 度数自适应融合 (GCN+MLP按度数加权)")
    print("=" * 70)
    t0 = time.time()

    data = load_classification_data(CLS_DATA)
    adj, features, labels = data["adj"], data["features"], data["labels"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    num_nodes, feat_dim = data["feat_dim"], data["feat_dim"]
    num_classes = data["num_classes"]

    # L2归一化
    features = normalize(features, norm="l2", axis=1).astype(np.float32)

    # 度数
    degree = np.array(adj.sum(axis=1)).flatten()
    isolated_mask = degree <= 1
    print(f"  孤立节点(度<=1): {isolated_mask.sum()}/{num_nodes} ({isolated_mask.mean():.1%})")

    # 邻接矩阵归一化
    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)

    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    train_t = torch.LongTensor(train_idx).to(device)
    test_t = torch.LongTensor(test_idx).to(device)
    degree_t = torch.FloatTensor(degree).to(device)

    # 90/10划分
    np.random.seed(42)
    perm = np.random.permutation(train_idx)
    n_val = int(len(train_idx) * 0.1)
    val_idx_arr = perm[:n_val]
    train_only_arr = perm[n_val:]
    val_t = torch.LongTensor(val_idx_arr).to(device)
    train_only_t = torch.LongTensor(train_only_arr).to(device)

    # ===== GCN集成 (处理有邻居节点) =====
    print(f"\n[GCN集成] {n_gcn}模型...")
    gcn_test_probs = []
    gcn_val_accs = []
    for i in range(n_gcn):
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
        gcn_test_probs.append(best_probs)
        gcn_val_accs.append(best_val)
        if (i + 1) % 5 == 0:
            print(f"    GCN {i+1}/{n_gcn} Val={best_val:.4f}")

    gcn_ensemble = np.mean(gcn_test_probs, axis=0)
    gcn_val_acc = np.mean(gcn_val_accs)
    print(f"  GCN集成 Val={gcn_val_acc:.4f}")

    # ===== MLP集成 (处理孤立节点, 不依赖图结构) =====
    print(f"\n[MLP集成] {n_mlp}模型 (Graph-MLP思路)...")
    adj_coo = adj.tocoo()
    neighbor_pairs = np.array(list(zip(adj_coo.row, adj_coo.col)))

    mlp_test_probs = []
    mlp_val_accs = []
    for i in range(n_mlp):
        seed = 200 + i * 10
        torch.manual_seed(seed); np.random.seed(seed)
        mlp = MLPCls(feat_dim, 512, num_classes, num_layers=3, dropout=0.5).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=0.005, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()
        best_val = 0; best_probs = None
        for epoch in range(1, 301):
            mlp.train(); opt.zero_grad()
            logits = mlp(features_t)
            ce_loss = crit(logits[train_only_t], labels_t[train_only_t])

            # NContrast对比损失
            if epoch % 2 == 0 and len(neighbor_pairs) > 0:
                emb = mlp.get_embedding(features_t)
                sample_idx = np.random.choice(len(neighbor_pairs), min(512, len(neighbor_pairs)), replace=False)
                pairs = neighbor_pairs[sample_idx]
                emb_a = emb[pairs[:, 0]]
                emb_b = emb[pairs[:, 1]]
                sim = (emb_a * emb_b).sum(1) / (emb_a.norm(dim=1) * emb_b.norm(dim=1) + 1e-8)
                contrast_loss = -sim.mean()
                loss = ce_loss + 0.1 * contrast_loss
            else:
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
            opt.step(); sched.step()
            if epoch % 10 == 0:
                mlp.eval()
                with torch.no_grad():
                    val_pred = mlp(features_t)[val_t].argmax(1)
                    val_acc = (val_pred == labels_t[val_t]).float().mean().item()
                    if val_acc > best_val:
                        best_val = val_acc
                        best_probs = F.softmax(mlp(features_t)[test_t], dim=1).cpu().numpy()
        mlp_test_probs.append(best_probs)
        mlp_val_accs.append(best_val)
        if (i + 1) % 5 == 0:
            print(f"    MLP {i+1}/{n_mlp} Val={best_val:.4f}")

    mlp_ensemble = np.mean(mlp_test_probs, axis=0)
    mlp_val_acc = np.mean(mlp_val_accs)
    print(f"  MLP集成 Val={mlp_val_acc:.4f}")

    # ===== 度数自适应融合 (核心创新) =====
    # 不是简单平均, 而是按度数加权:
    # 度数高 -> GCN权重高 (有邻居信息可用)
    # 度数低 -> MLP权重高 (不依赖图结构)
    print("\n[度数自适应融合]...")
    degree_test = degree[test_idx]
    # 权重: sigmoid(degree - 1), 度数1时约0.5, 度数5时约0.98
    gcn_weight = 1 / (1 + np.exp(-(degree_test - 1)))
    mlp_weight = 1 - gcn_weight

    # 归一化概率
    gcn_norm = gcn_ensemble / (gcn_ensemble.sum(1, keepdims=True) + 1e-8)
    mlp_norm = mlp_ensemble / (mlp_ensemble.sum(1, keepdims=True) + 1e-8)

    # 按度数加权融合
    fused = gcn_weight.reshape(-1, 1) * gcn_norm + mlp_weight.reshape(-1, 1) * mlp_norm

    # 验证集评估 (近似: 用GCN和MLP各自val_acc加权)
    val_degree = degree[val_idx_arr]
    val_gcn_w = 1 / (1 + np.exp(-(val_degree - 1)))
    adaptive_val = val_gcn_w.mean() * gcn_val_acc + (1 - val_gcn_w.mean()) * mlp_val_acc
    print(f"  自适应融合 Val={adaptive_val:.4f} (GCN={gcn_val_acc:.4f}, MLP={mlp_val_acc:.4f})")
    print(f"  孤立节点GCN权重均值={gcn_weight[degree_test<=1].mean():.3f}")

    # ===== 伪标签 =====
    print("[伪标签] 阈值>0.7...")
    conf = fused.max(axis=1)
    pseudo_mask = conf > 0.7
    pseudo_labels = fused.argmax(axis=1)
    final_train = np.concatenate([train_idx, test_idx[pseudo_mask]])
    final_labels = labels.copy()
    final_labels[test_idx[pseudo_mask]] = pseudo_labels[pseudo_mask]
    print(f"  伪标签: {pseudo_mask.sum()}个节点, 总训练: {len(final_train)}")

    # 用100%数据+伪标签重训GCN
    ft_mask = torch.LongTensor(final_train).to(device)
    ft_labels = torch.LongTensor(final_labels).to(device)
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
    pseudo_ensemble = np.mean(pseudo_probs, axis=0)

    # 标签传播0.4
    print("[标签传播] 0.4...")
    lp = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
    lp = lp / (lp.sum(1, keepdims=True) + 1e-8)
    final_pred = 0.6 * pseudo_ensemble + 0.4 * lp
    test_pred = final_pred.argmax(axis=1)

    elapsed = time.time() - t0
    print(f"\n[Task1完成] 度数自适应融合+伪标签+标签传播 | 耗时: {elapsed:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
        f.write("test_idx,label\n")
        for idx, p in zip(test_idx, test_pred):
            f.write(f"{idx},{p}\n")

    return adaptive_val


# ============================================================
# 推荐: SimRec损失 + 用户特征融合
# ============================================================
def compute_item_similarity(train_seqs, num_items):
    """计算物品共现相似度矩阵"""
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
    print(f"[SimRec] 相似度矩阵: {sim_dist.shape}")
    return torch.FloatTensor(sim_dist).to("cuda" if torch.cuda.is_available() else "cpu")


def run_recommendation_simrec(device="cuda", sim_lambda=0.3):
    """GRU4Rec + 用户特征 + SimRec损失"""
    print("\n" + "=" * 70)
    print(f"  Task2: GRU4Rec+用户特征 + SimRec损失(lambda={sim_lambda})")
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

    # 用户特征
    user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
    user_feat_dims = [int(user_df[col].max()) + 1 for col in user_feat_cols]
    user_feat_dict = {}
    for _, row in user_df.iterrows():
        user_feat_dict[row["uid"]] = torch.LongTensor([int(row[c]) for c in user_feat_cols])

    # 物品相似度
    sim_dist = compute_item_similarity(train_seqs, num_items)

    # 划分: 用短序列验证集 (匹配测试分布)
    np.random.seed(42)
    seq_lens = np.array([len(s) for s in train_seqs])
    # 验证集: 选短序列用户 (seq_len <= 10, 匹配测试分布)
    short_indices = np.where(seq_lens <= 10)[0]
    np.random.shuffle(short_indices)
    n_val = min(2000, len(short_indices))
    val_indices = short_indices[:n_val]
    val_set = set(val_indices.tolist())
    train_indices = np.array([i for i in range(len(train_seqs)) if i not in val_set])

    val_seqs = [train_seqs[i] for i in val_indices]
    val_targets = [train_targets[i] for i in val_indices]
    val_uids = [train_df.iloc[i]["uid"] for i in val_indices]
    tr_seqs = [train_seqs[i] for i in train_indices]
    tr_targets = [train_targets[i] for i in train_indices]
    tr_uids = [train_df.iloc[i]["uid"] for i in train_indices]

    print(f"  短序列验证集: {n_val}个用户 (seq_len<=10, 匹配测试分布)")
    print(f"  训练集: {len(tr_seqs)}")

    # 序列增强
    aug_seqs, aug_targets, aug_uids = list(tr_seqs), list(tr_targets), list(tr_uids)
    for seq, tgt, uid in zip(tr_seqs, tr_targets, tr_uids):
        if len(seq) > 5:
            for tl in [5, 10]:
                if len(seq) > tl:
                    aug_seqs.append(seq[-tl:]); aug_targets.append(tgt); aug_uids.append(uid)
    print(f"  序列增强: {len(tr_seqs)} -> {len(aug_seqs)}")

    # 构建带用户特征的模型
    model = build_recommendation_model(
        "gru4rec", num_items, embedding_dim=64, hidden_dim=128,
        num_layers=1, dropout=0.2, max_len=max_len,
        user_feat_dims=user_feat_dims).to(device)

    # 自定义Dataset (带用户特征)
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

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    # 验证函数 (需要用户特征)
    def evaluate_val():
        model.eval()
        ndcgs = []
        with torch.no_grad():
            for start in range(0, len(val_seqs), 256):
                batch_seqs = val_seqs[start:start+256]
                batch_targets = val_targets[start:start+256]
                batch_uids = val_uids[start:start+256]
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
                scores = model(seq_batch, length_batch, uf_batch)
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

    best_val = 0; best_state = None; patience = 0
    total_steps = len(train_loader) * 50
    warmup_steps = 1000
    global_step = 0

    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for seq_batch, length_batch, target_batch, uf_batch in train_loader:
            seq_batch = seq_batch.to(device)
            length_batch = length_batch.to(device)
            target_batch = target_batch.to(device)
            uf_batch = uf_batch.to(device)

            opt.zero_grad()
            scores = model(seq_batch, length_batch, uf_batch)
            scores[:, 0] = -1e9

            # CE损失
            ce_loss = crit(scores, target_batch)

            # SimRec相似性损失
            with torch.no_grad():
                target_sim = sim_dist[target_batch]
                target_sim[:, 0] = 0
                target_sim = target_sim / (target_sim.sum(1, keepdims=True) + 1e-8)
            log_probs = F.log_softmax(scores, dim=1)
            sim_loss = -(target_sim * log_probs).sum(1).mean()

            # lambda调度
            if global_step < warmup_steps:
                cur_lambda = sim_lambda * (global_step / warmup_steps)
            else:
                cur_lambda = sim_lambda * max(0.1, 1 - (global_step - warmup_steps) / (total_steps - warmup_steps))

            loss = (1 - cur_lambda) * ce_loss + cur_lambda * sim_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            total_loss += loss.item()
            global_step += 1

        val_ndcg = evaluate_val()
        print(f"  Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, Val NDCG@10={val_ndcg:.4f}, lambda={cur_lambda:.3f}")

        if val_ndcg > best_val:
            best_val = val_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 7:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    elapsed = time.time() - t0
    print(f"\n[Task2完成] Val NDCG@10={best_val:.4f} (短序列验证集) | 耗时: {elapsed:.1f}s")

    # 生成预测
    model.eval()
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
            scores = model(seq_batch, length_batch, uf_batch)
            scores[:, 0] = -1e9
            _, topk = scores.topk(10, dim=1)
            topk = topk.cpu().numpy()
            for pred in topk:
                items = [idx2item.get(i, "i000001") for i in pred if i in idx2item and i > 0]
                while len(items) < 10:
                    items.append("i000001")
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
    print("  AFAC2026 深度突破版")
    print("  分类: 度数自适应融合 (GCN+MLP按度数加权)")
    print("  推荐: SimRec损失+用户特征+短序列验证集")
    print("=" * 70)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_val = 0.69
    rec_val = 0
    if args.task in ("both", "cls"):
        cls_val = run_classification_adaptive(device=device, n_gcn=15, n_mlp=10)
    if args.task in ("both", "rec"):
        rec_val = run_recommendation_simrec(device=device, sim_lambda=0.3)

    elapsed = time.time() - t0
    final = 0.5 * cls_val + 0.5 * rec_val
    print(f"\n{'='*70}")
    print(f"  分类Val: {cls_val:.4f} | 推荐Val: {rec_val:.4f} | 预估: {final:.4f}")
    print(f"  总耗时: {elapsed/60:.1f}分钟")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
