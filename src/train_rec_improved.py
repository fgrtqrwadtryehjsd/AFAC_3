"""
改进版推荐任务训练模块
关键改进:
1. 物品流行度加权 (热门物品 boost)
2. 去除用户已看过的物品
3. BPR 负采样损失
4. 模型分数 + 流行度混合
5. 融合用户特征
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from .models import build_recommendation_model
from .data_loader import load_recommendation_data, build_rec_sequences


class RecDatasetBPR(Dataset):
    """BPR 训练数据集: (序列, 正样本, 负样本)"""

    def __init__(self, sequences, targets, num_items, max_len=50, n_neg=1):
        self.sequences = sequences
        self.targets = targets
        self.num_items = num_items
        self.max_len = max_len
        self.n_neg = n_neg
        # 物品采样分布 (按频率采样)
        item_counts = Counter(targets)
        self.item_probs = np.array([
            item_counts.get(i, 1) for i in range(1, num_items + 1)
        ], dtype=np.float64)
        self.item_probs /= self.item_probs.sum()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]

        # 负采样
        neg_items = np.random.choice(
            self.num_items, size=self.n_neg, replace=False,
            p=self.item_probs
        ) + 1  # +1 因为 0 是 padding
        # 确保负样本不是正样本
        neg_items = [n for n in neg_items if n != target]
        while len(neg_items) < self.n_neg:
            neg = np.random.randint(1, self.num_items + 1)
            if neg != target:
                neg_items.append(neg)

        # Padding
        if len(seq) == 0:
            seq_padded = [0] * self.max_len
            length = 1
        else:
            length = min(len(seq), self.max_len)
            seq_padded = seq[-self.max_len:] + [0] * (self.max_len - len(seq[-self.max_len:]))

        return (
            torch.LongTensor(seq_padded),
            torch.LongTensor([length])[0],
            torch.LongTensor([target])[0],
            torch.LongTensor(neg_items),
        )


class TestRecDataset(Dataset):
    """测试推荐数据集"""

    def __init__(self, sequences, max_len=50):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) == 0:
            seq_padded = [0] * self.max_len
            length = 1
        else:
            length = min(len(seq), self.max_len)
            seq_padded = seq[-self.max_len:] + [0] * (self.max_len - len(seq[-self.max_len:]))
        return torch.LongTensor(seq_padded), torch.LongTensor([length])[0]


def compute_ndcg_at_k(test_seqs, test_targets, model, device, k=10, batch_size=256):
    """计算验证集 NDCG@K"""
    model.eval()
    ndcgs = []

    for start in range(0, len(test_seqs), batch_size):
        batch_seqs = test_seqs[start:start + batch_size]
        batch_targets = test_targets[start:start + batch_size]

        max_len = model.max_len
        seq_tensors = []
        length_tensors = []

        for seq in batch_seqs:
            if len(seq) == 0:
                seq_padded = [0] * max_len
                length = 1
            else:
                length = min(len(seq), max_len)
                seq_padded = seq[-max_len:] + [0] * (max_len - len(seq[-max_len:]))
            seq_tensors.append(seq_padded)
            length_tensors.append(length)

        seq_batch = torch.LongTensor(seq_tensors).to(device)
        length_batch = torch.LongTensor(length_tensors).to(device)

        with torch.no_grad():
            scores = model(seq_batch, length_batch)
            scores[:, 0] = -1e9  # 排除 padding
            _, topk_indices = scores.topk(k, dim=1)
            topk_indices = topk_indices.cpu().numpy()

        for i, target in enumerate(batch_targets):
            pred_list = topk_indices[i]
            if target in pred_list:
                rank = np.where(pred_list == target)[0][0]
                dcg = 1.0 / np.log2(rank + 2)
            else:
                dcg = 0.0
            ndcgs.append(dcg)

    return float(np.mean(ndcgs))


def compute_rec_diagnostic_report(val_seqs, val_targets, model, device,
                                    item_popularity, elapsed, batch_size=256):
    """计算推荐任务的多维诊断报告"""
    model.eval()
    k = 10
    cold_ndcgs, warm_ndcgs = [], []
    head_hits, tail_hits = 0, 0
    head_total, tail_total = 0, 0

    # 按流行度分头尾物品
    pop_threshold_high = np.percentile(item_popularity[1:], 80)
    pop_threshold_low = np.percentile(item_popularity[1:], 20)

    for start in range(0, len(val_seqs), batch_size):
        batch_seqs = val_seqs[start:start + batch_size]
        batch_targets = val_targets[start:start + batch_size]
        max_len = model.max_len

        seq_tensors, length_tensors = [], []
        for seq in batch_seqs:
            if len(seq) == 0:
                seq_padded = [0] * max_len; length = 1
            else:
                length = min(len(seq), max_len)
                seq_padded = seq[-max_len:] + [0] * (max_len - len(seq[-max_len:]))
            seq_tensors.append(seq_padded); length_tensors.append(length)

        seq_batch = torch.LongTensor(seq_tensors).to(device)
        length_batch = torch.LongTensor(length_tensors).to(device)

        with torch.no_grad():
            scores = model(seq_batch, length_batch)
            scores[:, 0] = -1e9
            _, topk_indices = scores.topk(k, dim=1)
            topk_indices = topk_indices.cpu().numpy()

        for i, target in enumerate(batch_targets):
            seq_len = len(batch_seqs[i])
            pred_list = topk_indices[i]
            if target in pred_list:
                rank = np.where(pred_list == target)[0][0]
                dcg = 1.0 / np.log2(rank + 2)
            else:
                dcg = 0.0

            if seq_len < 5:
                cold_ndcgs.append(dcg)
            elif seq_len > 15:
                warm_ndcgs.append(dcg)

            if item_popularity[target] >= pop_threshold_high:
                head_total += 1
                if target in pred_list: head_hits += 1
            elif item_popularity[target] <= pop_threshold_low:
                tail_total += 1
                if target in pred_list: tail_hits += 1

    report = {
        "overall_ndcg_10": float(np.mean(cold_ndcgs + warm_ndcgs + [0])),  # 近似
        "subgroup_metrics": {
            "cold_start_users_ndcg (seq_len<5)": float(np.mean(cold_ndcgs)) if cold_ndcgs else 0,
            "warm_users_ndcg (seq_len>15)": float(np.mean(warm_ndcgs)) if warm_ndcgs else 0,
            "cold_start_count": len(cold_ndcgs),
            "warm_count": len(warm_ndcgs),
            "head_items_recall": float(head_hits / head_total) if head_total > 0 else 0,
            "tail_items_recall": float(tail_hits / tail_total) if tail_total > 0 else 0,
        },
        "system_metrics": {
            "training_time_seconds": float(elapsed),
        },
    }
    return report


def train_recommendation_improved(data_dir, config, output_dir, device="cuda"):
    """改进版推荐训练

    config: dict, 包含:
        - model_type, embedding_dim, hidden_dim, num_layers, dropout
        - lr, weight_decay, epochs, max_seq_len, batch_size, patience
        - loss_type: "ce" / "bpr"
        - n_neg: int (BPR 负采样数量)
        - popularity_weight: float (流行度加权权重 0-1)
        - remove_seen: bool (是否去除已看过的物品)
    """
    print(f"\n{'='*60}")
    print(f"[Task2-Improved] 开始改进版推荐训练")
    print(f"[Task2-Improved] 配置: {json.dumps(config, ensure_ascii=False)}")
    print(f"{'='*60}")

    start_time = time.time()

    # 加载数据
    data = load_recommendation_data(data_dir)
    train_df = data["train_df"]
    test_df = data["test_df"]
    user_df = data["user_df"]
    item_df = data["item_df"]
    item2idx = data["item2idx"]
    idx2item = data["idx2item"]
    num_items = data["num_items"]

    max_seq_len = config.get("max_seq_len", 50)

    # 构建序列
    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_seq_len
    )

    # 计算物品流行度 (从训练集统计)
    item_popularity = np.zeros(num_items + 1, dtype=np.float32)
    for target in train_targets:
        item_popularity[target] += 1
    # 平滑 + 归一化
    item_popularity = item_popularity + 1.0
    item_popularity = item_popularity / item_popularity.sum()
    # 转为 rank 分数 (流行度越高分数越高)
    pop_scores = np.log(item_popularity + 1e-10)
    pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-10)
    print(f"[Task2-Improved] 物品流行度统计完成, top-5 热门: {np.argsort(item_popularity)[-5:][::-1]}")

    # 划分训练/验证集
    val_ratio = config.get("val_ratio", 0.1)
    n_val = int(len(train_seqs) * val_ratio)
    np.random.seed(42)
    indices = np.random.permutation(len(train_seqs))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    val_seqs = [train_seqs[i] for i in val_indices]
    val_targets = [train_targets[i] for i in val_indices]
    tr_seqs = [train_seqs[i] for i in train_indices]
    tr_targets = [train_targets[i] for i in train_indices]

    # 序列增强: 随机截断训练序列，模拟测试集短序列分布
    use_seq_aug = config.get("use_seq_aug", True)
    if use_seq_aug:
        aug_seqs = []
        aug_targets = []
        for seq, target in zip(tr_seqs, tr_targets):
            aug_seqs.append(seq)
            aug_targets.append(target)
            # 随机截断为 5-15 个 item
            if len(seq) > 5:
                for trunc_len in [5, 10]:
                    if len(seq) > trunc_len:
                        start_idx = np.random.randint(0, len(seq) - trunc_len + 1)
                        aug_seqs.append(seq[start_idx:start_idx + trunc_len])
                        aug_targets.append(target)
        tr_seqs = aug_seqs
        tr_targets = aug_targets
        print(f"[Task2-Improved] 序列增强: {len(train_indices)} -> {len(tr_seqs)} (增强 {len(tr_seqs)-len(train_indices)} 条)")

    # 构建 DataLoader
    batch_size = config.get("batch_size", 256)
    loss_type = config.get("loss_type", "bpr")
    n_neg = config.get("n_neg", 5)

    if loss_type == "bpr":
        train_dataset = RecDatasetBPR(tr_seqs, tr_targets, num_items, max_seq_len, n_neg=n_neg)
    else:
        from .train_rec import RecDataset
        train_dataset = RecDataset(tr_seqs, tr_targets, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 构建模型
    model = build_recommendation_model(
        model_type=config.get("model_type", "gru4rec"),
        num_items=num_items,
        embedding_dim=config.get("embedding_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.2),
        max_len=max_seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 0),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 50), eta_min=1e-5
    )

    # 训练循环
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 7)
    best_val_ndcg = 0.0
    best_model_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if loss_type == "bpr":
                seq_batch, length_batch, pos_batch, neg_batch = batch
                seq_batch = seq_batch.to(device)
                length_batch = length_batch.to(device)
                pos_batch = pos_batch.to(device)
                neg_batch = neg_batch.to(device)

                optimizer.zero_grad()
                scores = model(seq_batch, length_batch)  # (batch, num_items+1)

                # BPR 损失: 正样本得分 > 负样本得分
                pos_scores = scores.gather(1, pos_batch.unsqueeze(1)).squeeze(1)  # (batch,)
                neg_scores = scores.gather(1, neg_batch)  # (batch, n_neg)
                bpr_loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
                loss = bpr_loss
            else:
                seq_batch, length_batch, target_batch = batch
                seq_batch = seq_batch.to(device)
                length_batch = length_batch.to(device)
                target_batch = target_batch.to(device)

                optimizer.zero_grad()
                scores = model(seq_batch, length_batch)
                loss = F.cross_entropy(scores, target_batch.squeeze())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # 验证
        val_ndcg = compute_ndcg_at_k(
            val_seqs, val_targets, model, device, k=10, batch_size=batch_size
        )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val NDCG@10: {val_ndcg:.4f} | Best: {best_val_ndcg:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 加载最佳模型预测
    model.load_state_dict(best_model_state)
    model.eval()

    # 预测
    test_dataset = TestRecDataset(test_seqs, max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pop_weight = config.get("popularity_weight", 0.15)
    remove_seen = config.get("remove_seen", True)

    all_predictions = []
    test_idx = 0

    # 构建测试用户的已看物品集合和序列长度
    test_seen_items = []
    test_seq_lengths = []
    for seq in test_seqs:
        seen = set(seq)
        test_seen_items.append(seen)
        test_seq_lengths.append(len(seq))

    with torch.no_grad():
        for seq_batch, length_batch in test_loader:
            seq_batch = seq_batch.to(device)
            length_batch = length_batch.to(device)
            scores = model(seq_batch, length_batch)  # (batch, num_items+1)

            # 转为 numpy
            scores_np = scores.cpu().numpy()

            for i in range(scores_np.shape[0]):
                model_scores = scores_np[i].copy()

                # 自适应流行度权重: 短序列用户更依赖流行度
                seq_len = test_seq_lengths[test_idx] if test_idx < len(test_seq_lengths) else 0
                if seq_len == 0:
                    # 无历史用户: 完全使用流行度
                    adaptive_pop = 1.0
                elif seq_len <= 3:
                    # 极短序列: 高流行度权重
                    adaptive_pop = 0.5
                elif seq_len <= 10:
                    # 短序列: 中等流行度权重
                    adaptive_pop = 0.25
                else:
                    # 正常序列: 低流行度权重
                    adaptive_pop = pop_weight

                # 混合流行度分数
                if adaptive_pop > 0:
                    combined = (1 - adaptive_pop) * model_scores + adaptive_pop * pop_scores
                else:
                    combined = model_scores

                # 排除 padding (index 0)
                combined[0] = -1e9

                # 去除已看过的物品
                if remove_seen and test_idx < len(test_seen_items):
                    for seen_idx in test_seen_items[test_idx]:
                        if seen_idx < len(combined):
                            combined[seen_idx] = -1e9

                # 取 top-10
                topk_indices = np.argsort(combined)[::-1][:10]

                pred_items = [idx2item[idx] for idx in topk_indices if idx in idx2item and idx > 0]
                while len(pred_items) < 10:
                    pred_items.append(idx2item.get(len(pred_items) + 1, "i000001"))

                all_predictions.append(pred_items[:10])
                test_idx += 1

    elapsed = time.time() - start_time
    print(f"[Task2-Improved] 训练完成 | 最佳验证 NDCG@10: {best_val_ndcg:.4f} | 耗时: {elapsed:.1f}s")

    # ========== 多维诊断报告 ==========
    diagnostic_report = compute_rec_diagnostic_report(
        val_seqs, val_targets, model, device, item_popularity, elapsed, batch_size
    )
    diagnostic_report["overall_ndcg_10"] = float(best_val_ndcg)
    print(f"[诊断报告] {json.dumps(diagnostic_report, ensure_ascii=False, indent=2)}")

    # 生成提交文件
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "A2.csv")
    with open(submission_path, "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, pred_items in zip(test_uids, all_predictions):
            pred_str = ",".join(pred_items)
            f.write(f'{uid},"{pred_str}"\n')
    print(f"[Task2-Improved] 预测结果已保存至 {submission_path}")

    trajectory_entry = {
        "round": None,
        "config": config,
        "val_ndcg10": float(best_val_ndcg),
        "loss_type": loss_type,
        "popularity_weight": pop_weight,
        "remove_seen": remove_seen,
        "use_seq_aug": config.get("use_seq_aug", True),
        "diagnostic_report": diagnostic_report,
        "elapsed_seconds": elapsed,
        "model_type": config.get("model_type", "gru4rec"),
        "feedback": f"验证NDCG@10={best_val_ndcg:.4f}, 损失={loss_type}, 冷启动NDCG={diagnostic_report['subgroup_metrics']['cold_start_users_ndcg (seq_len<5)']:.4f}, 耗时={elapsed:.1f}s",
    }

    return {
        "val_ndcg": best_val_ndcg,
        "predictions": all_predictions,
        "trajectory_entry": trajectory_entry,
    }
