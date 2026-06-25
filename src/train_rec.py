"""
Task2: 产品推荐任务训练模块
序列推荐: GRU4Rec / SASRec
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score

from .models import build_recommendation_model
from .data_loader import load_recommendation_data, build_rec_sequences


class RecDataset(Dataset):
    """推荐数据集"""

    def __init__(self, sequences, targets, max_len=50):
        self.sequences = sequences
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]

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

        return (
            torch.LongTensor(seq_padded),
            torch.LongTensor([length])[0],
        )


def compute_ndcg_at_k(test_seqs, test_targets, model, item2idx, idx2item,
                       device, k=10, batch_size=256):
    """计算验证集 NDCG@K"""
    model.eval()
    ndcgs = []

    # 分批处理
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
            scores = model(seq_batch, length_batch)  # (batch, num_items+1)
            # 取 top-K
            _, topk_indices = scores.topk(k, dim=1)
            topk_indices = topk_indices.cpu().numpy()

        for i, target in enumerate(batch_targets):
            pred_list = topk_indices[i]
            if target in pred_list:
                rank = np.where(pred_list == target)[0][0]
                dcg = 1.0 / np.log2(rank + 2)
            else:
                dcg = 0.0
            # IDCG for single relevant item is 1.0 (rank 0)
            ndcgs.append(dcg)

    return float(np.mean(ndcgs))


def train_recommendation(data_dir, config, output_dir, device="cuda"):
    """训练推荐模型并生成预测结果

    config: dict, 包含以下字段:
        - model_type: "gru4rec" / "sasrec"
        - embedding_dim: int
        - hidden_dim: int
        - num_layers: int
        - dropout: float
        - lr: float
        - weight_decay: float
        - epochs: int
        - max_seq_len: int
        - batch_size: int
        - val_ratio: float
    返回: dict, 包含 val_ndcg, predictions, trajectory_entry
    """
    print(f"\n{'='*60}")
    print(f"[Task2] 开始训练推荐模型")
    print(f"[Task2] 配置: {json.dumps(config, ensure_ascii=False)}")
    print(f"{'='*60}")

    start_time = time.time()

    # 加载数据
    data = load_recommendation_data(data_dir)
    train_df = data["train_df"]
    test_df = data["test_df"]
    item2idx = data["item2idx"]
    idx2item = data["idx2item"]
    num_items = data["num_items"]

    max_seq_len = config.get("max_seq_len", 50)

    # 构建序列
    train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
        train_df, test_df, item2idx, max_seq_len=max_seq_len
    )

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

    # 创建 DataLoader
    batch_size = config.get("batch_size", 256)
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

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 0),
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    epochs = config.get("epochs", 30)
    patience = config.get("patience", 5)
    best_val_ndcg = 0.0
    best_model_state = None
    no_improve_count = 0

    train_losses = []
    val_ndcgs = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for seq_batch, length_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(device)
            length_batch = length_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            scores = model(seq_batch, length_batch)  # (batch, num_items+1)
            loss = criterion(scores, target_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # 验证
        val_ndcg = compute_ndcg_at_k(
            val_seqs, val_targets, model, item2idx, idx2item,
            device, k=10, batch_size=batch_size
        )
        val_ndcgs.append(val_ndcg)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val NDCG@10: {val_ndcg:.4f} | Best: {best_val_ndcg:.4f}")

        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # 加载最佳模型进行预测
    model.load_state_dict(best_model_state)
    model.eval()

    # 对测试集进行预测
    test_dataset = TestRecDataset(test_seqs, max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_predictions = []
    with torch.no_grad():
        for seq_batch, length_batch in test_loader:
            seq_batch = seq_batch.to(device)
            length_batch = length_batch.to(device)
            scores = model(seq_batch, length_batch)
            # 取 top-10 (排除 padding item 0)
            scores[:, 0] = -1e9  # 排除 padding
            _, topk_indices = scores.topk(10, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            for pred_indices in topk_indices:
                pred_items = [idx2item[idx] for idx in pred_indices if idx in idx2item and idx > 0]
                # 如果不足10个，用热门item填充
                while len(pred_items) < 10:
                    # 用候选集中最热门的item填充
                    pred_items.append(idx2item.get(len(pred_items) + 1, "i000001"))
                all_predictions.append(pred_items[:10])

    elapsed = time.time() - start_time
    print(f"[Task2] 训练完成 | 最佳验证 NDCG@10: {best_val_ndcg:.4f} | 耗时: {elapsed:.1f}s")

    # 生成提交文件
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "A2.csv")
    with open(submission_path, "w", encoding="utf-8") as f:
        f.write("uid,prediction\n")
        for uid, pred_items in zip(test_uids, all_predictions):
            pred_str = ",".join(pred_items)
            f.write(f'{uid},"{pred_str}"\n')
    print(f"[Task2] 预测结果已保存至 {submission_path}")

    # 返回训练信息
    trajectory_entry = {
        "round": None,  # 由调用方设置
        "config": config,
        "val_ndcg10": float(best_val_ndcg),
        "train_loss": float(train_losses[-1]),
        "val_ndcg_history": [float(a) for a in val_ndcgs[-5:]],
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
        "model_type": config.get("model_type", "gru4rec"),
        "feedback": f"验证NDCG@10={best_val_ndcg:.4f}, 训练轮次={epoch}, 损失={train_losses[-1]:.4f}",
    }

    return {
        "val_ndcg": best_val_ndcg,
        "predictions": all_predictions,
        "trajectory_entry": trajectory_entry,
        "best_model_state": best_model_state,
    }
