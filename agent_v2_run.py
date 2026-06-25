"""
Agent V2: 自动化数据科学家 (Auto-DataScientist)
关键升级:
1. 多维诊断反馈: LLM 收到子群体指标 (冷启动/孤立节点/长尾物品)
2. 丰富动作空间: 模型架构/图工程/特征工程/数据增强/损失函数
3. ReAct 反思架构: 系统预设 + 记忆提炼 + 规划指令
4. 预算控制: 时间追踪 + STOP_AND_ENSEMBLE 动作
"""
import os
import sys
import json
import time
import argparse
import re
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.train_cls_improved import train_classification_improved
from src.train_rec_improved import train_recommendation_improved

CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"

# 总预算 (秒), 每个任务 120 分钟
BUDGET_SECONDS = 7200


class AutoDataScientistAgent:
    """自动化数据科学家 Agent: ReAct 反思 + 预算控制 + 多维诊断"""

    def __init__(self, api_key, model_name=LLM_MODEL, budget_rounds=3):
        self.client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
        self.model_name = model_name
        self.budget_rounds = budget_rounds
        self.memory = []          # 实验记忆 (含诊断报告)
        self.trajectory = []      # 完整轨迹
        self.cross_task_exp = []  # 跨任务经验
        self.start_time = None
        self.best_results = {}    # 各任务最佳结果

    def _llm_chat(self, system_prompt, user_prompt, temperature=0.7):
        """调用 LLM"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Agent] LLM 调用失败: {e}")
            return None

    def _extract_json(self, text):
        if not text:
            return None
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        return None

    def _get_remaining_time(self):
        """获取剩余预算时间 (分钟)"""
        if self.start_time is None:
            return BUDGET_SECONDS / 60
        elapsed = (time.time() - self.start_time) / 60
        return max(0, BUDGET_SECONDS / 60 - elapsed)

    def _build_memory_str(self):
        """构建实验记忆字符串 (含诊断报告和经验提炼)"""
        if not self.memory:
            return "暂无历史实验"
        lines = []
        for m in self.memory:
            lines.append(f"\n【第{m['round']}轮】")
            lines.append(f"  配置: {json.dumps(m.get('config', {}), ensure_ascii=False)[:200]}")
            if "val_accuracy" in m:
                lines.append(f"  结果: Val Acc = {m['val_accuracy']:.4f}")
            if "val_ndcg" in m:
                lines.append(f"  结果: Val NDCG@10 = {m['val_ndcg']:.4f}")
            # 诊断报告关键指标
            diag = m.get("diagnostic_report", {})
            if diag:
                sub = diag.get("subgroup_metrics", {})
                for k, v in sub.items():
                    if isinstance(v, (int, float)) and v != 0:
                        lines.append(f"  诊断 - {k}: {v:.4f}" if isinstance(v, float) else f"  诊断 - {k}: {v}")
            sys_m = diag.get("system_metrics", {})
            if sys_m:
                lines.append(f"  耗时: {sys_m.get('training_time_seconds', 0):.1f}s")
            lines.append(f"  策略: {m.get('strategy', 'N/A')}")
            lines.append(f"  理由: {m.get('rationale', 'N/A')[:150]}")
            if m.get("conclusion"):
                lines.append(f"  【经验提炼】{m['conclusion']}")
        return "\n".join(lines)

    def run_classification(self, device="cuda", n_ensemble=8):
        """分类任务: Agent 多轮 ReAct 实验"""
        print("\n" + "=" * 70)
        print("  Task1: 产品分类 - Auto-DataScientist Agent")
        print("=" * 70)

        self.start_time = time.time()
        self.memory = []
        self.trajectory = []

        system_prompt = """你是一位资深图学习与金融风控算法专家，正在一个受控的实验沙盒中工作。

## 任务信息
- 数据: 13,752 节点, 767 维特征, 10 类, 图极度稀疏 (平均度数 2.04)
- 训练节点 11,001, 测试节点 2,751
- 评测指标: Accuracy

## 可用的高阶动作 (不要只调超参数!)
1. model_architecture: gcn / sage / gat / appnp
2. graph_engineering: raw / drop_edge / knn_graph_fusion (KNN图补充稀疏节点的边)
3. feature_engineering: l2 / standard / none
4. label_strategy: none / pseudo_labeling (用模型预测做伪标签重训)
5. label_propagation: 0.0-0.5 (标签传播后处理权重)

## 超参数 (次要)
- hidden_dim: 128/256/512, num_layers: 2/3, dropout: 0.3/0.5/0.7
- lr: 0.005/0.01/0.02, weight_decay: 1e-4/5e-4/1e-3

## 决策原则
- 不要只做无脑调参! 要关注图结构增强、特征工程、模型范式切换
- 如果孤立节点(度数<=1)准确率很低, 考虑 KNN 图增强
- 如果训练损失很低但验证准确率不高, 考虑增加正则化或换模型
- 在预算有限时果断停止并集成历史最佳模型"""

        best_metric = 0.0
        best_round = 0

        for round_num in range(1, self.budget_rounds + 1):
            remaining = self._get_remaining_time()
            print(f"\n{'─'*60}")
            print(f"  分类任务 - 第 {round_num}/{self.budget_rounds} 轮 | 剩余预算: {remaining:.1f}min")
            print(f"{'─'*60}")

            memory_str = self._build_memory_str()
            cross_str = json.dumps(self.cross_task_exp, ensure_ascii=False)[:500] if self.cross_task_exp else "暂无"
            best_so_far = f"当前最佳: Val Acc = {best_metric:.4f}" if self.memory else "首轮实验"

            user_prompt = f"""## 当前状态
- 第 {round_num} 轮, 剩余预算 {remaining:.1f} 分钟
- {best_so_far}

## 历史实验记忆
{memory_str}

## 跨任务经验
{cross_str}

## 任务
请分析历史反馈中的【子群体指标】, 找出瓶颈, 并决定下一轮实验方向。
你必须在 rationale 中详细说明你是如何针对诊断报告中的问题进行改版的。

请输出 JSON:
```json
{{
    "rationale": "详细分析诊断报告, 说明本轮改进逻辑",
    "strategy": "exploration/exploitation/STOP_AND_ENSEMBLE",
    "config": {{
        "model_architecture": "gcn",
        "graph_engineering": "raw",
        "feature_engineering": "l2",
        "label_strategy": "none",
        "label_propagation": 0.3,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 0.0005,
        "epochs": 300,
        "patience": 50
    }}
}}
```

如果剩余时间不足或连续两轮无提升, 请设 strategy 为 "STOP_AND_ENSEMBLE"。"""

            response = self._llm_chat(system_prompt, user_prompt)
            decision = self._extract_json(response)

            if not decision:
                decision = {
                    "rationale": "默认配置",
                    "strategy": "exploration",
                    "config": {
                        "model_architecture": "gcn", "graph_engineering": "raw",
                        "feature_engineering": "l2", "label_strategy": "none",
                        "label_propagation": 0.3, "hidden_dim": 256, "num_layers": 2,
                        "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
                        "epochs": 300, "patience": 50
                    }
                }

            # 检查停止指令
            if decision.get("strategy") == "STOP_AND_ENSEMBLE":
                print(f"[Agent] 决定停止实验并集成历史最佳模型")
                break

            # 映射高阶动作到训练配置
            config = decision.get("config", {})
            config["model_type"] = config.pop("model_architecture", "gcn")
            config["feat_norm"] = config.pop("feature_engineering", "l2")
            config.setdefault("drop_edge_rate", 0.2 if config.get("graph_engineering") == "drop_edge" else 0.0)
            config.setdefault("normalization", "sym")
            config.setdefault("use_struct_feat", False)
            config.setdefault("n_ensemble", n_ensemble)

            print(f"[Agent] 图工程: {config.get('graph_engineering', 'raw')}")
            print(f"[Agent] 模型: {config.get('model_type')}, hidden={config.get('hidden_dim')}")
            print(f"[Agent] 策略: {decision.get('strategy')}")
            print(f"[Agent] 理由: {decision.get('rationale', '')[:200]}")

            # 运行实验
            round_output = os.path.join(OUTPUT_DIR, "task1", f"round{round_num}")
            result = train_classification_improved(
                npz_path=CLS_DATA_PATH, config=config, output_dir=round_output,
                device=device, n_ensemble=n_ensemble,
                use_label_prop=config.get("label_propagation", 0.3) > 0,
            )

            val_acc = result["val_accuracy"]

            # 构建记忆条目
            entry = {
                "round": round_num,
                "config": config,
                "val_accuracy": val_acc,
                "diagnostic_report": result["trajectory_entry"].get("diagnostic_report", {}),
                "rationale": decision.get("rationale", ""),
                "strategy": decision.get("strategy", ""),
            }

            # 经验提炼
            if self.memory:
                prev_acc = self.memory[-1]["val_accuracy"]
                if val_acc > prev_acc:
                    entry["conclusion"] = f"本轮提升 {val_acc - prev_acc:.4f}, 方向有效"
                else:
                    entry["conclusion"] = f"本轮下降 {prev_acc - val_acc:.4f}, 方向无效"

            self.memory.append(entry)

            traj = result["trajectory_entry"].copy()
            traj["round"] = round_num
            traj["rationale"] = decision.get("rationale", "")
            traj["strategy"] = decision.get("strategy", "")
            traj["conclusion"] = entry.get("conclusion", "")
            self.trajectory.append(traj)

            if val_acc > best_metric:
                best_metric = val_acc
                best_round = round_num
                import shutil
                shutil.copy2(os.path.join(round_output, "A1.csv"),
                            os.path.join(OUTPUT_DIR, "A1.csv"))

            print(f"\n[总结] 第 {round_num} 轮: Val Acc = {val_acc:.4f} | 最佳 = {best_metric:.4f}")

        # 保存轨迹
        traj_path = os.path.join(OUTPUT_DIR, "trajectory_B1.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_type": "classification",
                "total_rounds": len(self.trajectory),
                "experiments": self.trajectory,
                "best_result": max(self.memory, key=lambda x: x["val_accuracy"]) if self.memory else None,
            }, f, indent=2, ensure_ascii=False)

        # 提炼跨任务经验
        if self.memory:
            best = max(self.memory, key=lambda x: x["val_accuracy"])
            self.cross_task_exp.append({
                "task": "classification",
                "best_config": best["config"],
                "best_val_acc": best["val_accuracy"],
                "key_findings": " | ".join([m.get("conclusion", "") for m in self.memory if m.get("conclusion")]),
            })

        print(f"\n[Task1 完成] 最佳 Val Acc = {best_metric:.4f} (第 {best_round} 轮)")
        return best_metric

    def run_recommendation(self, device="cuda"):
        """推荐任务: Agent 多轮 ReAct 实验"""
        print("\n" + "=" * 70)
        print("  Task2: 产品推荐 - Auto-DataScientist Agent")
        print("=" * 70)

        self.memory = []
        self.trajectory = []

        system_prompt = """你是一位资深推荐系统与金融算法专家，正在一个受控的实验沙盒中工作。

## 任务信息
- 数据: 50,000 用户, 2,156 物品
- 训练用户 40,000 (平均序列长度 46), 测试用户 10,000 (平均序列长度 6.25)
- 评测指标: NDCG@10
- 核心挑战: 测试用户序列极短, 训练-测试分布不匹配

## 可用的高阶动作 (不要只调超参数!)
1. model_architecture: gru4rec / sasrec
2. data_augmentation: none / random_trunc / sliding_window (滑动窗口增强短序列)
3. loss_function: cross_entropy / bpr
4. side_features: none / user_features / item_features / both (融合用户/物品特征表)

## 超参数 (次要)
- embedding_dim: 32/64/128, hidden_dim: 64/128/256
- num_layers: 1/2, dropout: 0.1/0.2/0.3
- lr: 0.0005/0.001/0.005, max_seq_len: 20/50/100

## 决策原则
- 如果冷启动用户(seq_len<5) NDCG 很低, 考虑滑动窗口增强或融合用户特征
- 如果长尾物品召回不足, 考虑调整损失函数或增加负采样
- 不要只调 lr/hidden_dim, 要关注数据增强和特征融合
- 在预算有限时果断停止"""

        best_metric = 0.0
        best_round = 0

        for round_num in range(1, self.budget_rounds + 1):
            remaining = self._get_remaining_time()
            print(f"\n{'─'*60}")
            print(f"  推荐任务 - 第 {round_num}/{self.budget_rounds} 轮 | 剩余预算: {remaining:.1f}min")
            print(f"{'─'*60}")

            memory_str = self._build_memory_str()
            cross_str = json.dumps(self.cross_task_exp, ensure_ascii=False)[:500] if self.cross_task_exp else "暂无"
            best_so_far = f"当前最佳: Val NDCG = {best_metric:.4f}" if self.memory else "首轮实验"

            user_prompt = f"""## 当前状态
- 第 {round_num} 轮, 剩余预算 {remaining:.1f} 分钟
- {best_so_far}

## 历史实验记忆 (含诊断报告)
{memory_str}

## 跨任务经验 (来自分类任务)
{cross_str}

## 任务
请分析历史反馈中的【子群体指标】(冷启动用户NDCG、长尾物品召回率), 找出瓶颈, 决定下一轮方向。

请输出 JSON:
```json
{{
    "rationale": "详细分析诊断报告, 说明本轮改进逻辑",
    "strategy": "exploration/exploitation/STOP_AND_ENSEMBLE",
    "config": {{
        "model_architecture": "gru4rec",
        "data_augmentation": "random_trunc",
        "loss_function": "cross_entropy",
        "side_features": "none",
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.2,
        "lr": 0.001,
        "weight_decay": 0,
        "epochs": 50,
        "max_seq_len": 50,
        "batch_size": 256,
        "patience": 7
    }}
}}
```"""

            response = self._llm_chat(system_prompt, user_prompt)
            decision = self._extract_json(response)

            if not decision:
                decision = {
                    "rationale": "默认配置",
                    "strategy": "exploration",
                    "config": {
                        "model_architecture": "gru4rec", "data_augmentation": "random_trunc",
                        "loss_function": "cross_entropy", "side_features": "none",
                        "embedding_dim": 64, "hidden_dim": 128, "num_layers": 1,
                        "dropout": 0.2, "lr": 0.001, "weight_decay": 0,
                        "epochs": 50, "max_seq_len": 50, "batch_size": 256, "patience": 7
                    }
                }

            if decision.get("strategy") == "STOP_AND_ENSEMBLE":
                print(f"[Agent] 决定停止实验并集成历史最佳模型")
                break

            config = decision.get("config", {})
            config["model_type"] = config.pop("model_architecture", "gru4rec")
            config["loss_type"] = config.pop("loss_function", "cross_entropy")
            config["use_seq_aug"] = config.pop("data_augmentation", "random_trunc") != "none"
            config["popularity_weight"] = 0.0
            config["remove_seen"] = False
            config.setdefault("n_neg", 5)

            print(f"[Agent] 模型: {config.get('model_type')}, 数据增强: {config.get('use_seq_aug')}")
            print(f"[Agent] 策略: {decision.get('strategy')}")
            print(f"[Agent] 理由: {decision.get('rationale', '')[:200]}")

            round_output = os.path.join(OUTPUT_DIR, "task2", f"round{round_num}")
            result = train_recommendation_improved(
                data_dir=REC_DATA_DIR, config=config, output_dir=round_output, device=device,
            )

            val_ndcg = result["val_ndcg"]

            entry = {
                "round": round_num,
                "config": config,
                "val_ndcg": val_ndcg,
                "diagnostic_report": result["trajectory_entry"].get("diagnostic_report", {}),
                "rationale": decision.get("rationale", ""),
                "strategy": decision.get("strategy", ""),
            }

            if self.memory:
                prev_ndcg = self.memory[-1]["val_ndcg"]
                if val_ndcg > prev_ndcg:
                    entry["conclusion"] = f"本轮提升 {val_ndcg - prev_ndcg:.4f}, 方向有效"
                else:
                    entry["conclusion"] = f"本轮下降 {prev_ndcg - val_ndcg:.4f}, 方向无效"

            self.memory.append(entry)

            traj = result["trajectory_entry"].copy()
            traj["round"] = round_num
            traj["rationale"] = decision.get("rationale", "")
            traj["strategy"] = decision.get("strategy", "")
            traj["conclusion"] = entry.get("conclusion", "")
            self.trajectory.append(traj)

            if val_ndcg > best_metric:
                best_metric = val_ndcg
                best_round = round_num
                import shutil
                shutil.copy2(os.path.join(round_output, "A2.csv"),
                            os.path.join(OUTPUT_DIR, "A2.csv"))

            print(f"\n[总结] 第 {round_num} 轮: Val NDCG@10 = {val_ndcg:.4f} | 最佳 = {best_metric:.4f}")

        traj_path = os.path.join(OUTPUT_DIR, "trajectory_B2.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_type": "recommendation",
                "total_rounds": len(self.trajectory),
                "experiments": self.trajectory,
                "best_result": max(self.memory, key=lambda x: x["val_ndcg"]) if self.memory else None,
            }, f, indent=2, ensure_ascii=False)

        print(f"\n[Task2 完成] 最佳 Val NDCG@10 = {best_metric:.4f} (第 {best_round} 轮)")
        return best_metric


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 Agent V2 - Auto-DataScientist")
    parser.add_argument("--budget", type=int, default=3, help="每任务实验轮次")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_ensemble", type=int, default=8, help="分类集成数量")
    parser.add_argument("--task", type=str, default="both", choices=["both", "cls", "rec"])
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 Agent V2: Auto-DataScientist")
    print("  ReAct反思 + 多维诊断 + 预算控制 + STOP_AND_ENSEMBLE")
    print("=" * 70)

    agent = AutoDataScientistAgent(
        api_key=DEFAULT_API_KEY, budget_rounds=args.budget
    )
    start_time = time.time()

    cls_score = 0
    rec_score = 0

    if args.task in ("both", "cls"):
        cls_score = agent.run_classification(device=args.device, n_ensemble=args.n_ensemble)

    if args.task in ("both", "rec"):
        rec_score = agent.run_recommendation(device=args.device)

    elapsed = time.time() - start_time
    final = 0.5 * cls_score + 0.5 * rec_score

    print("\n" + "=" * 70)
    print("  最终结果")
    print("=" * 70)
    print(f"  分类: {cls_score:.4f}")
    print(f"  推荐: {rec_score:.4f}")
    print(f"  总分: {final:.4f}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
