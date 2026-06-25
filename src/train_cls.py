"""
Task1: 产品分类任务训练模块
图节点分类: GraphSAGE / GCN / GAT
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .models import build_classification_model
from .data_loader import (
    load_classification_data,
    preprocess_adj,
    sparse_csr_to_torch_sparse,
)


def train_classification(npz_path, config, output_dir, device="cuda"):
    """训练分类模型并生成预测结果

    config: dict, 包含以下字段:
        - model_type: "sage" / "gcn" / "gat"
        - hidden_dim: int
        - num_layers: int
        - dropout: float
        - lr: float
        - weight_decay: float
        - epochs: int
        - normalization: "sym" / "rw" / "none"
        - val_ratio: float (验证集比例)
    返回: dict, 包含 val_accuracy, best_config, predictions, trajectory_entry
    """
    print(f"\n{'='*60}")
    print(f"[Task1] 开始训练分类模型")
    print(f"[Task1] 配置: {json.dumps(config, ensure_ascii=False)}")
    print(f"{'='*60}")

    start_time = time.time()

    # 加载数据
    data = load_classification_data(npz_path)
    adj = data["adj"]
    features = data["features"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    num_nodes = data["num_nodes"]
    feat_dim = data["feat_dim"]
    num_classes = data["num_classes"]

    # 特征归一化 (L2 行归一化 + 标准化)
    feat_norm = config.get("feat_norm", "l2")
    if feat_norm == "l2":
        from sklearn.preprocessing import normalize
        features = normalize(features, norm="l2", axis=1)
        features = features.astype(np.float32)
    elif feat_norm == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        features = features.astype(np.float32)
    elif feat_norm == "none":
        pass  # 不做归一化

    # 预处理邻接矩阵
    normalization = config.get("normalization", "sym")
    add_self_loops = config.get("add_self_loops", True)
    adj_norm = preprocess_adj(adj, add_self_loops=add_self_loops, normalization=normalization)

    # 转换为 torch tensor
    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)

    # 划分训练/验证集 (从 train_idx 中划分)
    val_ratio = config.get("val_ratio", 0.1)
    train_labels = labels[train_idx]
    train_sub, val_sub = train_test_split(
        np.arange(len(train_idx)),
        test_size=val_ratio,
        random_state=42,
        stratify=train_labels if len(np.unique(train_labels)) > 1 else None,
    )
    train_mask = train_idx[train_sub]
    val_mask = train_idx[val_sub]

    train_mask_t = torch.LongTensor(train_mask).to(device)
    val_mask_t = torch.LongTensor(val_mask).to(device)
    test_idx_t = torch.LongTensor(test_idx).to(device)

    # 构建模型
    model = build_classification_model(
        model_type=config.get("model_type", "sage"),
        in_dim=feat_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_classes=num_classes,
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.5),
        normalization=normalization,
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.01),
        weight_decay=config.get("weight_decay", 5e-4),
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    epochs = config.get("epochs", 200)
    patience = config.get("patience", 30)
    best_val_acc = 0.0
    best_model_state = None
    no_improve_count = 0

    train_losses = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features_t, adj_sparse)
        loss = criterion(logits[train_mask_t], labels_t[train_mask_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            logits = model(features_t, adj_sparse)
            val_pred = logits[val_mask_t].argmax(dim=1).cpu().numpy()
            val_true = labels_t[val_mask_t].cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)

        train_losses.append(loss.item())
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # 加载最佳模型进行预测
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        logits = model(features_t, adj_sparse)
        test_pred = logits[test_idx_t].argmax(dim=1).cpu().numpy()

    elapsed = time.time() - start_time
    print(f"[Task1] 训练完成 | 最佳验证准确率: {best_val_acc:.4f} | 耗时: {elapsed:.1f}s")

    # 生成提交文件
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "A1.csv")
    with open(submission_path, "w") as f:
        f.write("test_idx,label\n")
        for idx, pred in zip(test_idx, test_pred):
            f.write(f"{idx},{pred}\n")
    print(f"[Task1] 预测结果已保存至 {submission_path}")

    # 返回训练信息
    trajectory_entry = {
        "round": None,  # 由调用方设置
        "config": config,
        "val_accuracy": float(best_val_acc),
        "train_loss": float(train_losses[-1]),
        "val_acc_history": [float(a) for a in val_accs[-10:]],
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
        "model_type": config.get("model_type", "sage"),
        "feedback": f"验证准确率={best_val_acc:.4f}, 训练轮次={epoch}, 损失={train_losses[-1]:.4f}",
    }

    return {
        "val_accuracy": best_val_acc,
        "predictions": test_pred,
        "trajectory_entry": trajectory_entry,
        "best_model_state": best_model_state,
    }
