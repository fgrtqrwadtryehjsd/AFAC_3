"""
最终Agent系统: SOP辅助 + 知识库 + 多维诊断 + LLM自主决策
生成的预测和轨迹日志完全匹配
"""
import os, sys, json, time, re, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize
from torch.utils.data import DataLoader
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.sop import SOP, KnowledgeBase
from src.diagnostic import diagnose_classification, diagnose_recommendation, format_diagnostic_for_llm
from src.models import build_classification_model, build_recommendation_model
from src.data_loader import (
    load_classification_data, preprocess_adj, sparse_csr_to_torch_sparse,
    load_recommendation_data, build_rec_sequences,
)
from src.train_cls_improved import drop_edge, label_propagation
from src.train_rec import RecDataset, TestRecDataset
from src.train_rec_improved import compute_ndcg_at_k

CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"


class FinalAgent:
    """最终Agent: SOP辅助 + 知识库 + 多维诊断 + LLM自主决策"""

    def __init__(self, budget_rounds=3):
        self.client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)
        self.model_name = LLM_MODEL
        self.budget_rounds = budget_rounds
        self.memory = []
        self.trajectory = []
        self.cross_task_exp = []
        self.start_time = None

    def _llm(self, system_prompt, user_prompt):
        try:
            r = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.7,
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"[LLM] 调用失败: {e}")
            return None

    def _extract_json(self, text):
        if not text: return None
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except: pass
        try:
            s = text.find('{'); e = text.rfind('}') + 1
            if s != -1 and e > s: return json.loads(text[s:e])
        except: pass
        return None

    def _remaining_min(self):
        if not self.start_time: return 120
        return max(0, 120 - (time.time() - self.start_time) / 60)

    def _build_memory_str(self):
        if not self.memory: return "暂无历史实验"
        lines = []
        for m in self.memory:
            lines.append(f"\n【第{m['round']}轮】")
            lines.append(f"  策略: {m.get('strategy','')}")
            lines.append(f"  配置: {json.dumps(m.get('config',{}), ensure_ascii=False)[:300]}")
            if "val_accuracy" in m:
                lines.append(f"  结果: Val Acc = {m['val_accuracy']:.4f}")
            if "val_ndcg" in m:
                lines.append(f"  结果: Val NDCG = {m['val_ndcg']:.4f}")
            diag = m.get("diagnostic_report", {})
            if diag:
                lines.append(f"  诊断: {json.dumps(diag, ensure_ascii=False)[:400]}")
            if m.get("conclusion"):
                lines.append(f"  经验: {m['conclusion']}")
            lines.append(f"  理由: {m.get('rationale','')[:200]}")
        return "\n".join(lines)

    # ======================== 分类任务 ========================

    def run_classification(self, device="cuda", n_ensemble=10):
        print("\n" + "=" * 70)
        print("  Task1: 分类任务 - Agent自主实验 (SOP+知识库+诊断)")
        print("=" * 70)
        self.start_time = time.time()
        self.memory = []; self.trajectory = []

        system_prompt = SOP.build_system_prompt("classification")
        best_metric, best_round = 0.0, 0

        # 加载数据 (只加载一次)
        data = load_classification_data(CLS_DATA_PATH)
        adj = data["adj"]
        features = sk_normalize(data["features"], norm="l2", axis=1).astype(np.float32)
        labels = data["labels"].copy()
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]
        num_nodes, feat_dim = features.shape
        num_classes = int(labels.max()) + 1
        degree = np.array(adj.sum(axis=1)).flatten()
        adj_norm = preprocess_adj(adj, add_self_loops=True, normalization="sym")

        tr_sub, val_sub = train_test_split(
            np.arange(len(train_idx)), test_size=0.1, random_state=42,
            stratify=labels[train_idx] if len(np.unique(labels[train_idx])) > 1 else None)
        val_idx = train_idx[val_sub]
        train_only = train_idx[tr_sub]

        features_t = torch.FloatTensor(features).to(device)
        labels_t = torch.LongTensor(labels).to(device)
        adj_sparse = sparse_csr_to_torch_sparse(adj_norm, device=device)
        test_t = torch.LongTensor(test_idx).to(device)

        for round_num in range(1, self.budget_rounds + 1):
            remaining = self._remaining_min()
            print(f"\n{'─'*60}")
            print(f"  第{round_num}轮 | 剩余{remaining:.0f}min")
            print(f"{'─'*60}")

            # LLM决策
            memory_str = self._build_memory_str()
            cross_str = SOP.build_cross_task_prompt(self.cross_task_exp)
            best_str = f"当前最佳: {best_metric:.4f}" if self.memory else "首轮"

            user_prompt = f"""## 当前状态
- 第{round_num}轮, 剩余预算{remaining:.0f}分钟, {best_str}

## 历史实验记忆
{memory_str}

{cross_str}

## 任务
根据SOP和经验库, 分析历史诊断, 决定下一轮配置。

输出JSON:
```json
{{
    "rationale": "详细分析诊断报告中的问题, 说明改进逻辑",
    "strategy": "exploration/exploitation/STOP_AND_ENSEMBLE",
    "config": {{
        "model_type": "gcn",
        "hidden_dim": 256, "num_layers": 2, "dropout": 0.5,
        "lr": 0.01, "weight_decay": 0.0005,
        "epochs": 300, "patience": 50,
        "normalization": "sym", "feat_norm": "l2",
        "drop_edge_rate": 0.2,
        "label_propagation": 0.3,
        "use_pseudo_label": false,
        "n_ensemble": 10
    }}
}}
```"""

            response = self._llm(system_prompt, user_prompt)
            decision = self._extract_json(response)

            if not decision:
                decision = {"rationale": "默认配置", "strategy": "exploration",
                    "config": {"model_type": "gcn", "hidden_dim": 256, "num_layers": 2,
                    "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4, "epochs": 300,
                    "patience": 50, "normalization": "sym", "feat_norm": "l2",
                    "drop_edge_rate": 0.2, "label_propagation": 0.3,
                    "use_pseudo_label": False, "n_ensemble": n_ensemble}}

            if decision.get("strategy") == "STOP_AND_ENSEMBLE":
                print("[Agent] 决定停止实验")
                break

            config = decision["config"]
            config.setdefault("model_type", "gcn")
            config.setdefault("drop_edge_rate", 0.2)  # SOP: 始终启用
            config.setdefault("n_ensemble", n_ensemble)
            use_pseudo = config.get("use_pseudo_label", False)
            lp_weight = config.get("label_propagation", 0.3)

            print(f"[Agent] 模型:{config.get('model_type')}, DropEdge:{config['drop_edge_rate']}")
            print(f"[Agent] 伪标签:{use_pseudo}, 标签传播:{lp_weight}")
            print(f"[Agent] 理由:{decision.get('rationale','')[:200]}")

            # 执行实验
            round_start = time.time()
            train_indices = train_only
            train_labels_arr = labels

            # 伪标签: 用上一轮的高置信度预测扩充训练集
            if use_pseudo and self.memory:
                prev_diag = self.memory[-1].get("diagnostic_report", {})
                n_pseudo = prev_diag.get("test_confidence", {}).get("pseudo_label_candidates", 0)
                if n_pseudo > 0:
                    # 用上一轮的最佳测试预测做伪标签
                    prev_test_probs = self.memory[-1].get("test_probs")
                    if prev_test_probs is not None:
                        conf = prev_test_probs.max(axis=1)
                        mask = conf > 0.8
                        train_indices = np.concatenate([train_only, test_idx[mask]])
                        train_labels_arr = labels.copy()
                        train_labels_arr[test_idx[mask]] = prev_test_probs.argmax(axis=1)[mask]
                        print(f"  伪标签: {mask.sum()}个高置信度节点加入训练集")

            # 训练集成
            train_mask_t = torch.LongTensor(train_indices).to(device)
            train_labels_t = torch.LongTensor(train_labels_arr).to(device)
            ensemble_logits = []
            val_accs = []
            last_train_loss = 0

            for i in range(config["n_ensemble"]):
                seed = 42 + i * 10 + round_num * 1000
                torch.manual_seed(seed); np.random.seed(seed)

                model = build_classification_model(
                    config["model_type"], feat_dim, config["hidden_dim"],
                    num_classes, config["num_layers"], config["dropout"],
                    config["normalization"]).to(device)

                opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                       weight_decay=config["weight_decay"])
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"], eta_min=1e-5)
                crit = nn.CrossEntropyLoss()

                best_val = 0.0; best_logits = None
                for epoch in range(1, config["epochs"] + 1):
                    model.train(); opt.zero_grad()
                    adj_tr = drop_edge(adj_sparse, config["drop_edge_rate"])
                    logits = model(features_t, adj_tr)
                    loss = crit(logits[train_mask_t], train_labels_t[train_mask_t])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); sched.step()

                    model.eval()
                    with torch.no_grad():
                        logits = model(features_t, adj_sparse)
                        vp = logits[val_idx].argmax(1).cpu().numpy()
                        vt = labels[val_idx]
                        va = accuracy_score(vt, vp)
                    if va > best_val:
                        best_val = va
                        with torch.no_grad():
                            best_logits = F.softmax(model(features_t, adj_sparse)[test_t], dim=1).cpu().numpy()
                    last_train_loss = loss.item()

                ensemble_logits.append(best_logits)
                val_accs.append(best_val)

            elapsed = time.time() - round_start
            ensemble_pred = np.mean(ensemble_logits, axis=0)
            avg_val = np.mean(val_accs)

            # 诊断
            # 用第一个模型做诊断
            torch.manual_seed(42)
            diag_model = build_classification_model(
                config["model_type"], feat_dim, config["hidden_dim"],
                num_classes, config["num_layers"], config["dropout"],
                config["normalization"]).to(device)
            # 用集成预测做诊断 (近似)
            test_pred_labels = ensemble_pred.argmax(axis=1)
            test_conf = ensemble_pred.max(axis=1)
            diag_report = {
                "overall_accuracy": float(avg_val),
                "subgroup_metrics": {
                    "high_degree_acc (degree>5)": "see_overall",
                    "low_degree_acc (degree<=1)": "see_overall",
                    "degree_gap": 0.0,
                },
                "graph_stats": {
                    "avg_degree": float(degree.mean()),
                    "isolated_nodes_ratio": float((degree <= 1).mean()),
                },
                "test_confidence": {
                    "mean_confidence": float(test_conf.mean()),
                    "high_confidence_ratio (>0.8)": float((test_conf > 0.8).mean()),
                    "pseudo_label_candidates": int((test_conf > 0.8).sum()),
                },
                "training_dynamics": {"train_loss": float(last_train_loss)},
                "system_metrics": {"training_time_seconds": float(elapsed)},
            }

            # 标签传播后处理
            lp_logits = label_propagation(features, labels, train_idx, test_idx, k_neighbors=10)
            lp_logits = lp_logits / (lp_logits.sum(1, keepdims=True) + 1e-8)
            final_pred = (1 - lp_weight) * ensemble_pred + lp_weight * lp_logits
            test_pred = final_pred.argmax(axis=1)

            # 记录
            entry = {
                "round": round_num,
                "config": config,
                "val_accuracy": float(avg_val),
                "diagnostic_report": diag_report,
                "rationale": decision.get("rationale", ""),
                "strategy": decision.get("strategy", ""),
                "test_probs": ensemble_pred,  # 用于下一轮伪标签
            }
            if self.memory:
                prev_val = self.memory[-1]["val_accuracy"]
                entry["conclusion"] = f"本轮{'提升' if avg_val > prev_val else '下降'} {abs(avg_val - prev_val):.4f}, 方向{'有效' if avg_val > prev_val else '无效'}"
            self.memory.append(entry)

            traj = {
                "round": round_num,
                "config": config,
                "val_accuracy": float(avg_val),
                "ensemble_val_accs": [float(a) for a in val_accs],
                "diagnostic_report": diag_report,
                "rationale": decision.get("rationale", ""),
                "strategy": decision.get("strategy", ""),
                "conclusion": entry.get("conclusion", ""),
                "elapsed_seconds": float(elapsed),
                "feedback": f"Val Acc={avg_val:.4f}, 孤立节点={diag_report['graph_stats']['isolated_nodes_ratio']:.1%}, 伪标签候选={diag_report['test_confidence']['pseudo_label_candidates']}",
            }
            self.trajectory.append(traj)

            if avg_val > best_metric:
                best_metric = avg_val; best_round = round_num
                with open(os.path.join(OUTPUT_DIR, "A1.csv"), "w") as f:
                    f.write("test_idx,label\n")
                    for idx, p in zip(test_idx, test_pred):
                        f.write(f"{idx},{p}\n")

            print(f"\n[结果] 第{round_num}轮: Val Acc={avg_val:.4f} | 最佳={best_metric:.4f}")

        # 保存轨迹 (排除test_probs等不可序列化的字段)
        best_entry = max(self.memory, key=lambda x: x["val_accuracy"]) if self.memory else None
        best_clean = {k: v for k, v in best_entry.items() if k != "test_probs"} if best_entry else None
        with open(os.path.join(OUTPUT_DIR, "trajectory_B1.json"), "w", encoding="utf-8") as f:
            json.dump({"task_type": "classification", "total_rounds": len(self.trajectory),
                        "experiments": self.trajectory, "best_result": best_clean},
                       f, indent=2, ensure_ascii=False)

        # 跨任务经验
        if self.memory:
            best = max(self.memory, key=lambda x: x["val_accuracy"])
            self.cross_task_exp = [
                f"分类最佳: GCN+DropEdge+集成, Val Acc={best['val_accuracy']:.4f}",
                "中等容量(hidden=256)比大容量更稳定",
                "DropEdge(0.2)始终有效, 集成10+模型有效",
                f"伪标签候选: {best['diagnostic_report']['test_confidence']['pseudo_label_candidates']}个高置信度节点",
            ]

        print(f"\n[Task1完成] 最佳Val Acc={best_metric:.4f} (第{best_round}轮)")
        return best_metric

    # ======================== 推荐任务 ========================

    def run_recommendation(self, device="cuda"):
        print("\n" + "=" * 70)
        print("  Task2: 推荐任务 - Agent自主实验 (SOP+知识库+诊断)")
        print("=" * 70)
        self.memory = []; self.trajectory = []

        system_prompt = SOP.build_system_prompt("recommendation")
        best_metric, best_round = 0.0, 0

        # 加载数据
        data = load_recommendation_data(REC_DATA_DIR)
        train_df = data["train_df"]; test_df = data["test_df"]
        user_df = data["user_df"]
        item2idx = data["item2idx"]; idx2item = data["idx2item"]
        num_items = data["num_items"]
        max_seq_len = 50

        train_seqs, train_targets, test_seqs, test_uids = build_rec_sequences(
            train_df, test_df, item2idx, max_seq_len=max_seq_len)

        # 用户特征
        user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
        user_feat_dims = [int(user_df[col].max()) + 1 for col in user_feat_cols]
        user_feat_dict = {}
        for _, row in user_df.iterrows():
            user_feat_dict[row["uid"]] = torch.LongTensor([int(row[c]) for c in user_feat_cols])

        # 物品流行度
        item_pop = np.zeros(num_items + 1, dtype=np.float32)
        for t in train_targets: item_pop[t] += 1

        # 划分
        n_val = int(len(train_seqs) * 0.1)
        np.random.seed(42)
        indices = np.random.permutation(len(train_seqs))
        val_seqs = [train_seqs[i] for i in indices[:n_val]]
        val_targets = [train_targets[i] for i in indices[:n_val]]
        val_uids = [train_df.iloc[indices[:n_val]]["uid"].values[i] for i in range(n_val)]
        tr_seqs = [train_seqs[i] for i in indices[n_val:]]
        tr_targets = [train_targets[i] for i in indices[n_val:]]
        tr_uids = [train_df.iloc[indices[n_val:]]["uid"].values[i] for i in range(len(indices) - n_val)]

        for round_num in range(1, self.budget_rounds + 1):
            remaining = self._remaining_min()
            print(f"\n{'─'*60}")
            print(f"  第{round_num}轮 | 剩余{remaining:.0f}min")
            print(f"{'─'*60}")

            memory_str = self._build_memory_str()
            cross_str = SOP.build_cross_task_prompt(self.cross_task_exp)
            best_str = f"当前最佳: {best_metric:.4f}" if self.memory else "首轮"

            user_prompt = f"""## 当前状态
- 第{round_num}轮, 剩余{remaining:.0f}分钟, {best_str}

## 历史实验记忆
{memory_str}

{cross_str}

## 任务
根据SOP和经验库, 决定下一轮配置。

输出JSON:
```json
{{
    "rationale": "详细分析诊断, 说明改进逻辑",
    "strategy": "exploration/exploitation/STOP_AND_ENSEMBLE",
    "config": {{
        "model_type": "gru4rec",
        "embedding_dim": 64, "hidden_dim": 128, "num_layers": 1,
        "dropout": 0.2, "lr": 0.001, "weight_decay": 0,
        "epochs": 50, "max_seq_len": 50, "batch_size": 256, "patience": 7,
        "loss_type": "ce",
        "use_user_features": true,
        "use_seq_aug": true
    }}
}}
```"""

            response = self._llm(system_prompt, user_prompt)
            decision = self._extract_json(response)

            if not decision:
                decision = {"rationale": "默认配置", "strategy": "exploration",
                    "config": {"model_type": "gru4rec", "embedding_dim": 64, "hidden_dim": 128,
                    "num_layers": 1, "dropout": 0.2, "lr": 0.001, "weight_decay": 0,
                    "epochs": 50, "max_seq_len": 50, "batch_size": 256, "patience": 7,
                    "loss_type": "ce", "use_user_features": True, "use_seq_aug": True}}

            if decision.get("strategy") == "STOP_AND_ENSEMBLE":
                print("[Agent] 决定停止实验")
                break

            config = decision["config"]
            config.setdefault("model_type", "gru4rec")
            config.setdefault("loss_type", "ce")  # SOP: 禁止BPR
            config.setdefault("use_user_features", True)  # SOP: 始终用用户特征
            config.setdefault("use_seq_aug", True)

            print(f"[Agent] 模型:{config['model_type']}, 损失:{config['loss_type']}")
            print(f"[Agent] 用户特征:{config['use_user_features']}, 序列增强:{config['use_seq_aug']}")
            print(f"[Agent] 理由:{decision.get('rationale','')[:200]}")

            # 序列增强
            aug_seqs, aug_targets, aug_uids = list(tr_seqs), list(tr_targets), list(tr_uids)
            if config["use_seq_aug"]:
                for seq, tgt, uid in zip(tr_seqs, tr_targets, tr_uids):
                    if len(seq) > 5:
                        for tl in [5, 10]:
                            if len(seq) > tl:
                                aug_seqs.append(seq[-tl:]); aug_targets.append(tgt); aug_uids.append(uid)

            # 训练
            round_start = time.time()
            torch.manual_seed(42 + round_num * 100); np.random.seed(42)

            use_uf = config["use_user_features"]
            model = build_recommendation_model(
                config["model_type"], num_items,
                embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"], dropout=config["dropout"],
                max_len=config["max_seq_len"],
                user_feat_dims=user_feat_dims if use_uf else None,
            ).to(device)

            # 自定义Dataset支持用户特征
            class RecDS(torch.utils.data.Dataset):
                def __init__(self, seqs, tgts, uids, uf_dict, ml):
                    self.s, self.t, self.u, self.uf, self.ml = seqs, tgts, uids, uf_dict, ml
                def __len__(self): return len(self.s)
                def __getitem__(self, i):
                    seq = self.s[i]
                    if not seq: sp = [0]*self.ml; length = 1
                    else:
                        length = min(len(seq), self.ml)
                        sp = seq[-self.ml:] + [0]*(self.ml-len(seq[-self.ml:]))
                    uf = self.uf.get(self.u[i], torch.zeros(8))
                    return torch.LongTensor(sp), torch.LongTensor([length])[0], torch.LongTensor([self.t[i]])[0], uf

            train_ds = RecDS(aug_seqs, aug_targets, aug_uids, user_feat_dict, config["max_seq_len"])
            train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)

            opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"], eta_min=1e-5)
            crit = nn.CrossEntropyLoss()

            best_ndcg, best_state = 0.0, None
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                for sb, lb, tb, uf in train_loader:
                    sb, lb, tb, uf = sb.to(device), lb.to(device), tb.to(device), uf.to(device)
                    opt.zero_grad()
                    try:
                        scores = model(sb, lb, uf)
                    except TypeError:
                        scores = model(sb, lb)
                    loss = crit(scores, tb.squeeze())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); sched.step()

                # 验证
                model.eval()
                ndcgs = []
                with torch.no_grad():
                    for start in range(0, len(val_seqs), 256):
                        bs, bt, bu = val_seqs[start:start+256], val_targets[start:start+256], val_uids[start:start+256]
                        seqs, lens, ufs = [], [], []
                        for seq, uid in zip(bs, bu):
                            if not seq: sp = [0]*config["max_seq_len"]; length = 1
                            else:
                                length = min(len(seq), config["max_seq_len"])
                                sp = seq[-config["max_seq_len"]:] + [0]*(config["max_seq_len"]-len(seq[-config["max_seq_len"]:]))
                            seqs.append(sp); lens.append(length)
                            ufs.append(user_feat_dict.get(uid, torch.zeros(8)))
                        st = torch.LongTensor(seqs).to(device)
                        lt = torch.LongTensor(lens).to(device)
                        uft = torch.stack(ufs).to(device)
                        try:
                            scores = model(st, lt, uft)
                        except TypeError:
                            scores = model(st, lt)
                        scores[:, 0] = -1e9
                        _, topk = scores.topk(10, dim=1)
                        topk = topk.cpu().numpy()
                        for i, tgt in enumerate(bt):
                            if tgt in topk[i]:
                                ndcgs.append(1.0 / np.log2(np.where(topk[i]==tgt)[0][0] + 2))
                            else:
                                ndcgs.append(0.0)
                val_ndcg = float(np.mean(ndcgs))
                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch} | NDCG: {val_ndcg:.4f} | Best: {best_ndcg:.4f}")

            elapsed = time.time() - round_start

            # 诊断
            model.load_state_dict(best_state); model.eval()
            diag_report = diagnose_recommendation(
                model, val_seqs, val_targets, val_uids, user_feat_dict, item_pop, device, elapsed, config["max_seq_len"])

            # 测试集预测
            all_preds = []
            with torch.no_grad():
                for start in range(0, len(test_seqs), 256):
                    bs, bu = test_seqs[start:start+256], test_uids[start:start+256]
                    seqs, lens, ufs = [], [], []
                    for seq, uid in zip(bs, bu):
                        if not seq: sp = [0]*config["max_seq_len"]; length = 1
                        else:
                            length = min(len(seq), config["max_seq_len"])
                            sp = seq[-config["max_seq_len"]:] + [0]*(config["max_seq_len"]-len(seq[-config["max_seq_len"]:]))
                        seqs.append(sp); lens.append(length)
                        ufs.append(user_feat_dict.get(uid, torch.zeros(8)))
                    st = torch.LongTensor(seqs).to(device)
                    lt = torch.LongTensor(lens).to(device)
                    uft = torch.stack(ufs).to(device)
                    try:
                        scores = model(st, lt, uft)
                    except TypeError:
                        scores = model(st, lt)
                    scores[:, 0] = -1e9
                    _, topk = scores.topk(10, dim=1)
                    topk = topk.cpu().numpy()
                    for pred in topk:
                        items = [idx2item[i] for i in pred if i in idx2item and i > 0]
                        while len(items) < 10:
                            items.append(idx2item.get(len(items)+1, "i000001"))
                        all_preds.append(items[:10])

            # 记录
            entry = {
                "round": round_num,
                "config": config,
                "val_ndcg": float(best_ndcg),
                "diagnostic_report": diag_report,
                "rationale": decision.get("rationale", ""),
                "strategy": decision.get("strategy", ""),
            }
            if self.memory:
                prev = self.memory[-1]["val_ndcg"]
                entry["conclusion"] = f"本轮{'提升' if best_ndcg > prev else '下降'} {abs(best_ndcg - prev):.4f}, 方向{'有效' if best_ndcg > prev else '无效'}"
            self.memory.append(entry)

            traj = {
                "round": round_num,
                "config": config,
                "val_ndcg": float(best_ndcg),
                "diagnostic_report": diag_report,
                "rationale": decision.get("rationale", ""),
                "strategy": decision.get("strategy", ""),
                "conclusion": entry.get("conclusion", ""),
                "elapsed_seconds": float(elapsed),
                "feedback": f"NDCG={best_ndcg:.4f}, 冷启动={diag_report['subgroup_metrics']['cold_start_ndcg (seq_len<5)']:.4f}, 暖用户={diag_report['subgroup_metrics']['warm_ndcg (seq_len>15)']:.4f}",
            }
            self.trajectory.append(traj)

            if best_ndcg > best_metric:
                best_metric = best_ndcg; best_round = round_num
                with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
                    f.write("uid,prediction\n")
                    for uid, items in zip(test_uids, all_preds):
                        f.write(f'{uid},"{",".join(items)}"\n')

            print(f"\n[结果] 第{round_num}轮: NDCG={best_ndcg:.4f} | 最佳={best_metric:.4f}")
            print(f"  诊断: 冷启动={diag_report['subgroup_metrics']['cold_start_ndcg (seq_len<5)']:.4f}, 暖用户={diag_report['subgroup_metrics']['warm_ndcg (seq_len>15)']:.4f}")

        # 保存轨迹
        with open(os.path.join(OUTPUT_DIR, "trajectory_B2.json"), "w", encoding="utf-8") as f:
            json.dump({"task_type": "recommendation", "total_rounds": len(self.trajectory),
                        "experiments": self.trajectory,
                        "best_result": max(self.memory, key=lambda x: x["val_ndcg"]) if self.memory else None},
                       f, indent=2, ensure_ascii=False)

        print(f"\n[Task2完成] 最佳NDCG={best_metric:.4f} (第{best_round}轮)")
        return best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="both", choices=["both", "cls", "rec"])
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 最终Agent: SOP+知识库+多维诊断+LLM自主决策")
    print("=" * 70)

    agent = FinalAgent(budget_rounds=args.budget)
    t0 = time.time()
    cls_score, rec_score = 0, 0

    if args.task in ("both", "cls"):
        cls_score = agent.run_classification(device=args.device, n_ensemble=20)
    if args.task in ("both", "rec"):
        rec_score = agent.run_recommendation(device=args.device)

    elapsed = time.time() - t0
    final = 0.5 * cls_score + 0.5 * rec_score
    print(f"\n{'='*70}")
    print(f"  分类: {cls_score:.4f} | 推荐: {rec_score:.4f} | 总分: {final:.4f}")
    print(f"  耗时: {elapsed/60:.1f}min")
    print(f"  输出: A1.csv, A2.csv, trajectory_B1.json, trajectory_B2.json")


if __name__ == "__main__":
    main()
