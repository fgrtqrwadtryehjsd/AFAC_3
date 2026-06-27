"""
诊断引擎: 计算多维反馈报告
分类: 按度数分组的子群体准确率
推荐: 按序列长度分组的子群体NDCG
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def diagnose_classification(model, features_t, adj_sparse, labels_t,
                             train_idx, val_idx, test_idx_t, degree, train_loss, elapsed):
    """分类任务多维诊断报告"""
    model.eval()
    with torch.no_grad():
        logits = model(features_t, adj_sparse)
        val_pred = logits[val_idx].argmax(dim=1).cpu().numpy()
        val_true = labels_t[val_idx].cpu().numpy()
        test_probs = F.softmax(logits[test_idx_t], dim=1).cpu().numpy()

    overall_acc = accuracy_score(val_true, val_pred)

    # 按度数分组
    val_degree = degree[val_idx]
    high_mask = val_degree > 5
    low_mask = val_degree <= 1

    high_acc = accuracy_score(val_true[high_mask], val_pred[high_mask]) if high_mask.sum() > 0 else 0
    low_acc = accuracy_score(val_true[low_mask], val_pred[low_mask]) if low_mask.sum() > 0 else 0

    # 测试集置信度分布
    test_confidence = test_probs.max(axis=1)
    high_conf_ratio = (test_confidence > 0.8).mean()

    report = {
        "overall_accuracy": float(overall_acc),
        "subgroup_metrics": {
            "high_degree_acc (degree>5)": float(high_acc),
            "low_degree_acc (degree<=1)": float(low_acc),
            "high_degree_count": int(high_mask.sum()),
            "low_degree_count": int(low_mask.sum()),
            "degree_gap": float(high_acc - low_acc),
        },
        "graph_stats": {
            "avg_degree": float(degree.mean()),
            "isolated_nodes_ratio": float((degree <= 1).mean()),
        },
        "test_confidence": {
            "mean_confidence": float(test_confidence.mean()),
            "high_confidence_ratio (>0.8)": float(high_conf_ratio),
            "pseudo_label_candidates": int((test_confidence > 0.8).sum()),
        },
        "training_dynamics": {
            "train_loss": float(train_loss),
        },
        "system_metrics": {
            "training_time_seconds": float(elapsed),
        },
    }
    return report, test_probs


def diagnose_recommendation(model, val_seqs, val_targets, val_uids,
                              user_feat_dict, item_popularity, device, elapsed, max_len=50):
    """推荐任务多维诊断报告"""
    model.eval()
    ndcgs = []
    cold_ndcgs, warm_ndcgs = [], []
    head_hits, tail_hits = 0, 0
    head_total, tail_total = 0, 0

    pop_high = np.percentile(item_popularity[1:], 80)
    pop_low = np.percentile(item_popularity[1:], 20)

    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(val_seqs), batch_size):
            batch_seqs = val_seqs[start:start + batch_size]
            batch_targets = val_targets[start:start + batch_size]
            batch_uids = val_uids[start:start + batch_size]

            seqs, lens, ufs = [], [], []
            for seq, uid in zip(batch_seqs, batch_uids):
                if len(seq) == 0:
                    sp = [0] * max_len; length = 1
                else:
                    length = min(len(seq), max_len)
                    sp = seq[-max_len:] + [0] * (max_len - len(seq[-max_len:]))
                seqs.append(sp); lens.append(length)
                ufs.append(user_feat_dict.get(uid, torch.zeros(8)))

            seq_t = torch.LongTensor(seqs).to(device)
            len_t = torch.LongTensor(lens).to(device)
            uf_t = torch.stack(ufs).to(device)

            # 尝试带用户特征的前向传播
            try:
                scores = model(seq_t, len_t, uf_t)
            except TypeError:
                scores = model(seq_t, len_t)

            scores[:, 0] = -1e9
            _, topk = scores.topk(10, dim=1)
            topk = topk.cpu().numpy()

            for i, tgt in enumerate(batch_targets):
                seq_len = len(batch_seqs[i])
                if tgt in topk[i]:
                    rank = np.where(topk[i] == tgt)[0][0]
                    dcg = 1.0 / np.log2(rank + 2)
                else:
                    dcg = 0.0
                ndcgs.append(dcg)

                if seq_len < 5:
                    cold_ndcgs.append(dcg)
                elif seq_len > 15:
                    warm_ndcgs.append(dcg)

                if item_popularity[tgt] >= pop_high:
                    head_total += 1
                    if tgt in topk[i]: head_hits += 1
                elif item_popularity[tgt] <= pop_low:
                    tail_total += 1
                    if tgt in topk[i]: tail_hits += 1

    overall = float(np.mean(ndcgs)) if ndcgs else 0
    cold = float(np.mean(cold_ndcgs)) if cold_ndcgs else 0
    warm = float(np.mean(warm_ndcgs)) if warm_ndcgs else 0

    report = {
        "overall_ndcg_10": overall,
        "subgroup_metrics": {
            "cold_start_ndcg (seq_len<5)": cold,
            "warm_ndcg (seq_len>15)": warm,
            "cold_warm_gap": float(warm - cold),
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


def format_diagnostic_for_llm(report, task_type):
    """将诊断报告格式化为LLM易读的文本"""
    lines = ["## 诊断报告"]
    if task_type == "classification":
        lines.append(f"整体准确率: {report['overall_accuracy']:.4f}")
        sub = report["subgroup_metrics"]
        lines.append(f"高度数节点准确率(degree>5): {sub['high_degree_acc (degree>5)']:.4f} ({sub['high_degree_count']}个)")
        lines.append(f"低度数节点准确率(degree<=1): {sub['low_degree_acc (degree<=1)']:.4f} ({sub['low_degree_count']}个)")
        lines.append(f"度数差距: {sub['degree_gap']:.4f}")
        gs = report["graph_stats"]
        lines.append(f"平均度数: {gs['avg_degree']:.2f}, 孤立节点比例: {gs['isolated_nodes_ratio']:.1%}")
        tc = report["test_confidence"]
        lines.append(f"测试集高置信度(>0.8)比例: {tc['high_confidence_ratio (>0.8)']:.1%}, 可做伪标签: {tc['pseudo_label_candidates']}个")
        lines.append(f"训练损失: {report['training_dynamics']['train_loss']:.4f}")
    else:
        lines.append(f"整体NDCG@10: {report['overall_ndcg_10']:.4f}")
        sub = report["subgroup_metrics"]
        lines.append(f"冷启动用户NDCG(seq_len<5): {sub['cold_start_ndcg (seq_len<5)']:.4f} ({sub['cold_start_count']}个)")
        lines.append(f"暖用户NDCG(seq_len>15): {sub['warm_ndcg (seq_len>15)']:.4f} ({sub['warm_count']}个)")
        lines.append(f"冷启动差距: {sub['cold_warm_gap']:.4f}")
        lines.append(f"头部物品召回: {sub['head_items_recall']:.4f}, 长尾物品召回: {sub['tail_items_recall']:.4f}")
    lines.append(f"耗时: {report['system_metrics']['training_time_seconds']:.1f}s")
    return "\n".join(lines)
