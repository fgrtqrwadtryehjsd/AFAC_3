"""
改进版主运行脚本
使用最优配置 + 集成学习 + 后处理优化
"""
import os
import sys
import json
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.train_cls_improved import train_classification_improved
from src.train_rec_improved import train_recommendation_improved

# 数据路径
CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 最优配置 (基于实验分析)
CLS_CONFIG = {
    "model_type": "gcn",          # GCN 对 transductive 任务更优
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 300,
    "patience": 50,
    "normalization": "sym",       # 对称归一化
    "feat_norm": "l2",            # L2 行归一化
    "drop_edge_rate": 0.2,        # DropEdge 20%
    "label_prop_alpha": 0.3,      # 标签传播混合权重
    "use_struct_feat": False,     # 度数特征未带来提升，关闭
}

REC_CONFIG = {
    "model_type": "gru4rec",
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 0,
    "epochs": 50,
    "max_seq_len": 50,
    "batch_size": 256,
    "patience": 7,
    "loss_type": "ce",            # CE 损失
    "n_neg": 5,
    "popularity_weight": 0.0,     # 关闭流行度混合
    "remove_seen": False,         # 关闭去重
    "use_seq_aug": True,          # 序列增强: 随机截断模拟短序列
}


def run_improved_classification(device="cuda", n_ensemble=5):
    """运行改进版分类任务"""
    print("\n" + "=" * 70)
    print("  Task1-Improved: 产品分类 (GCN + DropEdge + 集成 + 标签传播)")
    print("=" * 70)

    result = train_classification_improved(
        npz_path=CLS_DATA_PATH,
        config=CLS_CONFIG,
        output_dir=OUTPUT_DIR,
        device=device,
        n_ensemble=n_ensemble,
        use_label_prop=True,
    )

    return result


def run_improved_recommendation(device="cuda"):
    """运行改进版推荐任务"""
    print("\n" + "=" * 70)
    print("  Task2-Improved: 产品推荐 (GRU4Rec + BPR + 流行度 + 去重)")
    print("=" * 70)

    result = train_recommendation_improved(
        data_dir=REC_DATA_DIR,
        config=REC_CONFIG,
        output_dir=OUTPUT_DIR,
        device=device,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 改进版实验")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="both", choices=["both", "cls", "rec"])
    parser.add_argument("--n_ensemble", type=int, default=5, help="分类任务集成数量")
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 赛题三: 改进版自动化实验")
    print("=" * 70)
    print(f"  分类配置: {json.dumps(CLS_CONFIG, ensure_ascii=False)}")
    print(f"  推荐配置: {json.dumps(REC_CONFIG, ensure_ascii=False)}")
    print(f"  集成数量: {args.n_ensemble}")

    start_time = time.time()

    cls_result = None
    rec_result = None

    if args.task in ("both", "cls"):
        cls_result = run_improved_classification(
            device=args.device, n_ensemble=args.n_ensemble
        )

    if args.task in ("both", "rec"):
        rec_result = run_improved_recommendation(
            device=args.device
        )

    # 打印结果摘要
    print("\n" + "=" * 70)
    print("  最终结果摘要")
    print("=" * 70)

    cls_score = 0
    rec_score = 0

    if cls_result:
        cls_score = cls_result["val_accuracy"]
        print(f"  分类任务: Val Accuracy = {cls_score:.4f}")
    if rec_result:
        rec_score = rec_result["val_ndcg"]
        print(f"  推荐任务: Val NDCG@10 = {rec_score:.4f}")

    final_score = 0.5 * cls_score + 0.5 * rec_score
    print(f"\n  预估最终分数: 0.5 * {cls_score:.4f} + 0.5 * {rec_score:.4f} = {final_score:.4f}")

    elapsed = time.time() - start_time
    print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"\n  输出文件:")
    print(f"    - {os.path.join(OUTPUT_DIR, 'A1.csv')}")
    print(f"    - {os.path.join(OUTPUT_DIR, 'A2.csv')}")


if __name__ == "__main__":
    main()
