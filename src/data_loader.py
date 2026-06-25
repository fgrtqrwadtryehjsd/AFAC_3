"""
数据加载模块
- Task1: 加载 .npz 图节点分类数据
- Task2: 加载 CSV 序列推荐数据
"""
import os
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch


# ======================== Task1: 图节点分类数据 ========================

def load_classification_data(npz_path):
    """加载 .npz 格式的图节点分类数据，返回稠密特征矩阵、稀疏邻接矩阵、标签等"""
    data = np.load(npz_path)

    # 还原邻接矩阵 (CSR)
    adj = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )

    # 还原特征矩阵 (CSR -> dense)
    features = sp.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    ).toarray().astype(np.float32)

    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    num_nodes, feat_dim = features.shape
    num_classes = int(labels.max()) + 1

    print(f"[分类数据] 节点数={num_nodes}, 特征维度={feat_dim}, 类别数={num_classes}")
    print(f"           训练节点={len(train_idx)}, 测试节点={len(test_idx)}")
    print(f"           邻接矩阵非零元素={adj.nnz}, 平均度数={adj.nnz / num_nodes:.2f}")

    return {
        "adj": adj,
        "features": features,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "num_nodes": num_nodes,
        "feat_dim": feat_dim,
        "num_classes": num_classes,
    }


def preprocess_adj(adj, add_self_loops=True, normalization="sym"):
    """预处理邻接矩阵: 添加自环 + 归一化
    normalization: "sym" (对称归一化), "rw" (随机游走归一化), "none"
    """
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], format="csr")

    if normalization == "none":
        return adj

    # 计算度矩阵
    deg = np.array(adj.sum(axis=1)).flatten()

    if normalization == "sym":
        deg_inv_sqrt = np.zeros_like(deg, dtype=np.float64)
        mask = deg > 0
        deg_inv_sqrt[mask] = np.power(deg[mask], -0.5)
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    elif normalization == "rw":
        deg_inv = np.zeros_like(deg, dtype=np.float64)
        mask = deg > 0
        deg_inv[mask] = np.power(deg[mask], -1)
        D_inv = sp.diags(deg_inv)
        adj_norm = D_inv @ adj
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return adj_norm.tocsr()


def sparse_csr_to_torch_sparse(csr_mat, device="cuda"):
    """将 scipy CSR 稀疏矩阵转为 torch sparse COO 张量"""
    coo = csr_mat.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


# ======================== Task2: 序列推荐数据 ========================

def load_recommendation_data(data_dir):
    """加载序列推荐 CSV 数据"""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    user_df = pd.read_csv(os.path.join(data_dir, "user.csv"))
    item_df = pd.read_csv(os.path.join(data_dir, "item.csv"))

    # 解析 item 序列
    def parse_seq(seq_str):
        if pd.isna(seq_str) or seq_str == "":
            return []
        return seq_str.split(",")

    train_df["item_seq"] = train_df["item_seq_dedup"].apply(parse_seq)
    test_df["item_seq"] = test_df["item_seq_dedup"].apply(parse_seq)

    # 构建 item id 映射
    all_items = sorted(item_df["iid"].unique())
    item2idx = {iid: idx + 1 for idx, iid in enumerate(all_items)}  # 0 留给 padding
    idx2item = {v: k for k, v in item2idx.items()}

    num_items = len(all_items)
    num_users = len(user_df)
    num_train = len(train_df)
    num_test = len(test_df)

    print(f"[推荐数据] 用户数={num_users}, 物品数={num_items}")
    print(f"           训练样本={num_train}, 测试样本={num_test}")
    print(f"           平均序列长度(训练)={train_df['item_seq'].apply(len).mean():.2f}")
    print(f"           平均序列长度(测试)={test_df['item_seq'].apply(len).mean():.2f}")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "user_df": user_df,
        "item_df": item_df,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "num_items": num_items,
        "num_users": num_users,
        "num_train": num_train,
        "num_test": num_test,
    }


def build_rec_sequences(train_df, test_df, item2idx, max_seq_len=50):
    """构建训练/测试的序列数据"""
    # 训练数据: (序列, 目标item)
    train_seqs = []
    train_targets = []
    for _, row in train_df.iterrows():
        seq = [item2idx[i] for i in row["item_seq"] if i in item2idx]
        target = item2idx.get(row["target_iid"], 0)
        if target == 0:
            continue
        # 截取最近 max_seq_len 个
        seq = seq[-max_seq_len:]
        train_seqs.append(seq)
        train_targets.append(target)

    # 测试数据: (序列, uid)
    test_seqs = []
    test_uids = []
    for _, row in test_df.iterrows():
        seq = [item2idx[i] for i in row["item_seq"] if i in item2idx]
        seq = seq[-max_seq_len:]
        test_seqs.append(seq)
        test_uids.append(row["uid"])

    return train_seqs, train_targets, test_seqs, test_uids
