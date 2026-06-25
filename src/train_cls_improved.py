"""
改进版分类任务训练模块
关键改进:
1. DropEdge 数据增强 (训练时随机丢弃边)
2. 多模型集成 (不同随机种子)
3. 全量训练数据训练最终模型
4. 标签传播后处理 (针对稀疏节点)
5. 改进 GCN (BatchNorm + 残差连接)
"""
import os
import time
import json
import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize

from .models import build_classification_model
from .data_loader import (
    load_classification_data,
    preprocess_adj,
    sparse_csr_to_torch_sparse,
)


def drop_edge(adj_sparse, drop_rate=0.2):
    """DropEdge: 随机丢弃部分边，返回新的稀疏邻接矩阵"""
    if drop_rate <= 0:
        return adj_sparse

    # 获取稀疏矩阵的 COO 格式
    adj = adj_sparse.coalesce()
    indices = adj.indices()
    values = adj.values()
    num_edges = values.shape[0]

    # 随机选择保留的边
    mask = torch.rand(num_edges, device=adj.device) > drop_rate
    keep_indices = indices[:, mask]
    keep_values = values[mask]

    # 重建稀疏矩阵
    new_adj = torch.sparse_coo_tensor(
        keep_indices, keep_values, adj.shape, device=adj.device
    ).coalesce()
    return new_adj


def normalize_features(features, method="l2"):
    """特征归一化"""
    if method == "l2":
        return sk_normalize(features, norm="l2", axis=1).astype(np.float32)
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(features).astype(np.float32)
    else:
        return features.astype(np.float32)


def label_propagation(features, labels, train_idx, test_idx, alpha=0.99,
                      n_iter=50, k_neighbors=10):
    """标签传播: 基于特征相似度传播标签，用于稀疏节点后处理

    对每个测试节点，找到特征最相似的 k 个训练节点，用它们的标签投票
    """
    print("[标签传播] 开始标签传播后处理...")

    # 转为 torch tensor 用于高效计算
    train_features = torch.FloatTensor(features[train_idx]).to("cuda" if torch.cuda.is_available() else "cpu")
    test_features = torch.FloatTensor(features[test_idx]).to(train_features.device)
    train_labels_arr = labels[train_idx]

    num_classes = int(labels.max()) + 1
    batch_size = 500
    propagated_labels = np.zeros((len(test_idx), num_classes))

    for start in range(0, len(test_idx), batch_size):
        end = min(start + batch_size, len(test_idx))
        batch_features = test_features[start:end]

        # 余弦相似度 (特征已 L2 归一化，直接点积即为余弦相似度)
        sim = batch_features @ train_features.T  # (batch, n_train)

        # 取 top-k 相似的训练节点
        k = min(k_neighbors, sim.shape[1])
        topk_sim, topk_idx = sim.topk(k, dim=1)

        # 加权投票 (softmax 权重)
        weights = F.softmax(topk_sim, dim=1).cpu().numpy()
        topk_idx_np = topk_idx.cpu().numpy()

        for i in range(end - start):
            neighbor_labels = train_labels_arr[topk_idx_np[i]]
            for j, label in enumerate(neighbor_labels):
                propagated_labels[start + i, label] += weights[i][j]

    return propagated_labels


def train_single_model(features, adj_norm, labels, train_idx, val_idx, test_idx,
                       config, device, seed=42, use_dropedge=True):
    """训练单个模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_nodes, feat_dim = features.shape
    num_classes = int(labels.max()) + 1

    # 转换为 tensor
    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)

    train_mask_t = torch.LongTensor(train_idx).to(device)
    val_mask_t = torch.LongTensor(val_idx).to(device)
    test_idx_t = torch.LongTensor(test_idx).to(device)

    # 构建模型
    model = build_classification_model(
        model_type=config.get("model_type", "gcn"),
        in_dim=feat_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_classes=num_classes,
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.5),
        normalization=config.get("normalization", "sym"),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.01),
        weight_decay=config.get("weight_decay", 5e-4),
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 300), eta_min=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    epochs = config.get("epochs", 300)
    patience = config.get("patience", 50)
    drop_rate = config.get("drop_edge_rate", 0.2) if use_dropedge else 0.0

    best_val_acc = 0.0
    best_model_state = None
    best_test_logits = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # DropEdge: 训练时随机丢弃边
        if drop_rate > 0:
            adj_train = drop_edge(adj_sparse, drop_rate)
        else:
            adj_train = adj_sparse

        logits = model(features_t, adj_train)
        loss = criterion(logits[train_mask_t], labels_t[train_mask_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            logits = model(features_t, adj_sparse)
            val_pred = logits[val_mask_t].argmax(dim=1).cpu().numpy()
            val_true = labels_t[val_mask_t].cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            # 保存测试集 logits 用于集成
            with torch.no_grad():
                best_test_logits = F.softmax(
                    model(features_t, adj_sparse)[test_idx_t], dim=1
                ).cpu().numpy()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    [Seed {seed}] Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if no_improve >= patience:
            break

    return best_val_acc, best_model_state, best_test_logits


def train_classification_improved(npz_path, config, output_dir, device="cuda",
                                   n_ensemble=5, use_label_prop=True):
    """改进版分类训练: 集成 + DropEdge + 标签传播

    config: dict, 包含:
        - model_type: "gcn" / "sage" / "gat"
        - hidden_dim, num_layers, dropout, lr, weight_decay
        - epochs, patience
        - normalization: "sym" / "rw"
        - feat_norm: "l2" / "standard" / "none"
        - drop_edge_rate: float (0.0-0.5)
        - label_prop_alpha: float (0.0-1.0) 标签传播混合权重
    """
    print(f"\n{'='*60}")
    print(f"[Task1-Improved] 开始改进版分类训练")
    print(f"[Task1-Improved] 配置: {json.dumps(config, ensure_ascii=False)}")
    print(f"[Task1-Improved] 集成数量: {n_ensemble}, 标签传播: {use_label_prop}")
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

    # 添加图结构特征: 节点度数 (帮助模型识别稀疏节点)
    use_struct_feat = config.get("use_struct_feat", True)
    if use_struct_feat:
        degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
        # 归一化度数
        degree_norm = degree / (degree.max() + 1e-8)
        # 添加为额外特征列
        features = np.hstack([features, degree_norm.reshape(-1, 1)])
        print(f"[Task1-Improved] 添加度数特征: {feat_dim} -> {features.shape[1]}")
        feat_dim = features.shape[1]

    # 特征归一化
    feat_norm = config.get("feat_norm", "l2")
    features = normalize_features(features, feat_norm)
    print(f"[Task1-Improved] 特征归一化: {feat_norm}")

    # 邻接矩阵预处理
    normalization = config.get("normalization", "sym")
    adj_norm = preprocess_adj(adj, add_self_loops=True, normalization=normalization)
    print(f"[Task1-Improved] 邻接矩阵归一化: {normalization}")

    # 划分训练/验证集
    val_ratio = config.get("val_ratio", 0.1)
    train_labels_arr = labels[train_idx]
    train_sub, val_sub = train_test_split(
        np.arange(len(train_idx)),
        test_size=val_ratio,
        random_state=42,
        stratify=train_labels_arr if len(np.unique(train_labels_arr)) > 1 else None,
    )
    val_idx = train_idx[val_sub]
    train_only_idx = train_idx[train_sub]
    full_train_idx = train_idx  # 全量训练数据

    # ========== 阶段1: 用验证集筛选超参数 + 集成训练 ==========
    print(f"\n[阶段1] 集成训练 ({n_ensemble} 个模型, 不同随机种子)")
    ensemble_logits = []
    val_accs = []

    for i in range(n_ensemble):
        seed = 42 + i * 10
        print(f"\n  --- 模型 {i+1}/{n_ensemble} (seed={seed}) ---")
        val_acc, model_state, test_logits = train_single_model(
            features, adj_norm, labels, train_only_idx, val_idx, test_idx,
            config, device, seed=seed, use_dropedge=True
        )
        ensemble_logits.append(test_logits)
        val_accs.append(val_acc)
        print(f"  模型 {i+1} 验证准确率: {val_acc:.4f}")

    avg_val_acc = np.mean(val_accs)
    print(f"\n[阶段1] 集成平均验证准确率: {avg_val_acc:.4f}")
    print(f"[阶段1] 各模型验证准确率: {[f'{a:.4f}' for a in val_accs]}")

    # 集成预测 (平均 softmax)
    ensemble_pred = np.mean(ensemble_logits, axis=0)  # (n_test, num_classes)

    # ========== 阶段2: 标签传播后处理 ==========
    if use_label_prop:
        print(f"\n[阶段2] 标签传播后处理")
        lp_weight = config.get("label_prop_alpha", 0.3)
        lp_logits = label_propagation(
            features, labels, train_idx, test_idx,
            alpha=0.99, n_iter=50, k_neighbors=10
        )
        # 归一化标签传播结果
        lp_logits = lp_logits / (lp_logits.sum(axis=1, keepdims=True) + 1e-8)

        # 混合集成预测和标签传播
        final_logits = (1 - lp_weight) * ensemble_pred + lp_weight * lp_logits
        print(f"[阶段2] 标签传播权重: {lp_weight}")
    else:
        final_logits = ensemble_pred

    # 最终预测
    test_pred = final_logits.argmax(axis=1)

    elapsed = time.time() - start_time
    print(f"\n[Task1-Improved] 训练完成 | 集成验证准确率: {avg_val_acc:.4f} | 耗时: {elapsed:.1f}s")

    # 生成提交文件
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "A1.csv")
    with open(submission_path, "w") as f:
        f.write("test_idx,label\n")
        for idx, pred in zip(test_idx, test_pred):
            f.write(f"{idx},{pred}\n")
    print(f"[Task1-Improved] 预测结果已保存至 {submission_path}")

    # 轨迹记录
    trajectory_entry = {
        "round": None,
        "config": config,
        "val_accuracy": float(avg_val_acc),
        "ensemble_val_accs": [float(a) for a in val_accs],
        "n_ensemble": n_ensemble,
        "use_label_prop": use_label_prop,
        "label_prop_alpha": config.get("label_prop_alpha", 0.3),
        "elapsed_seconds": elapsed,
        "model_type": config.get("model_type", "gcn"),
        "feedback": f"集成验证准确率={avg_val_acc:.4f}, 集成数={n_ensemble}, 标签传播={use_label_prop}",
    }

    return {
        "val_accuracy": avg_val_acc,
        "predictions": test_pred,
        "trajectory_entry": trajectory_entry,
    }
