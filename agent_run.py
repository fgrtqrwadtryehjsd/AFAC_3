"""
Agent 主运行脚本
- LLM 驱动的多轮实验决策
- 实验记忆维护 + 反馈分析
- 轨迹日志记录 (trajectory_B1.json, trajectory_B2.json)
- 跨任务经验迁移
"""
import os
import sys
import json
import time
import argparse
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


class AFACAgent:
    """自动化实验 Agent: LLM 决策 + 多轮迭代 + 实验记忆"""

    def __init__(self, api_key, model_name=LLM_MODEL, budget=3):
        self.client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
        self.model_name = model_name
        self.budget = budget
        self.memory = []          # 实验记忆
        self.trajectory = []      # 完整轨迹
        self.cross_task_exp = []  # 跨任务经验

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
        """从文本中提取 JSON"""
        import re
        if not text:
            return None
        # 尝试代码块
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        # 尝试直接提取
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        return None

    def run_classification(self, device="cuda", n_ensemble=10):
        """运行分类任务的 Agent 多轮实验"""
        print("\n" + "=" * 70)
        print("  Task1: 产品分类 - Agent 自动化实验")
        print("=" * 70)

        system_prompt = """你是 AFAC2026 自动化实验 Agent，负责图节点分类任务的实验优化。

## 任务信息
- 数据: 13,752 节点, 767 维特征, 10 类, 图极度稀疏 (平均度数 2.04)
- 训练节点 11,001, 测试节点 2,751
- 评测指标: Accuracy
- 模型: GCN (BatchNorm + 残差连接 + DropEdge)
- 后处理: 标签传播 (基于特征相似度)

## 可调参数
- hidden_dim: 128/256/512
- num_layers: 2/3
- dropout: 0.3/0.5/0.7
- lr: 0.005/0.01/0.02
- weight_decay: 1e-4/5e-4/1e-3
- drop_edge_rate: 0.1/0.2/0.3
- label_prop_alpha: 0.1/0.2/0.3/0.4
- use_struct_feat: true/false (添加度数特征)

## 决策原则
- 第1轮: 稳健 baseline 配置
- 第2轮: 根据反馈探索不同方向
- 第3轮: 基于最佳方向精细调优"""

        best_metric = 0.0
        best_result = None

        for round_num in range(1, self.budget + 1):
            print(f"\n{'─'*60}")
            print(f"  分类任务 - 第 {round_num}/{self.budget} 轮")
            print(f"{'─'*60}")

            # LLM 决策
            history_str = json.dumps(self.memory, indent=2, ensure_ascii=False) if self.memory else "暂无历史"
            cross_str = json.dumps(self.cross_task_exp, indent=2, ensure_ascii=False) if self.cross_task_exp else "暂无"

            user_prompt = f"""## 当前状态
- 第 {round_num} 轮, 剩余预算 {self.budget - round_num + 1}
- 已完成实验: {len(self.memory)} 次

## 历史实验
{history_str}

## 跨任务经验
{cross_str}

请分析历史反馈，输出下一轮实验配置 JSON:
```json
{{
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 0.0005,
    "epochs": 300,
    "patience": 50,
    "normalization": "sym",
    "feat_norm": "l2",
    "drop_edge_rate": 0.2,
    "label_prop_alpha": 0.3,
    "use_struct_feat": true,
    "strategy": "exploration/exploitation",
    "rationale": "决策理由"
}}
```"""

            response = self._llm_chat(system_prompt, user_prompt)
            config = self._extract_json(response)

            if not config:
                # 默认配置
                config = {
                    "hidden_dim": 256, "num_layers": 2, "dropout": 0.5,
                    "lr": 0.01, "weight_decay": 5e-4, "epochs": 300,
                    "patience": 50, "normalization": "sym", "feat_norm": "l2",
                    "drop_edge_rate": 0.2, "label_prop_alpha": 0.3,
                    "use_struct_feat": True, "strategy": "default",
                    "rationale": "默认配置"
                }

            config.setdefault("model_type", "gcn")
            config.setdefault("hidden_dim", 256)
            config.setdefault("num_layers", 2)
            config.setdefault("dropout", 0.5)
            config.setdefault("lr", 0.01)
            config.setdefault("weight_decay", 5e-4)
            config.setdefault("epochs", 300)
            config.setdefault("patience", 50)
            config.setdefault("normalization", "sym")
            config.setdefault("feat_norm", "l2")
            config.setdefault("drop_edge_rate", 0.2)
            config.setdefault("label_prop_alpha", 0.3)
            config.setdefault("use_struct_feat", True)

            print(f"[Agent] 配置: hidden={config['hidden_dim']}, layers={config['num_layers']}, "
                  f"dropout={config['dropout']}, lr={config['lr']}")
            print(f"[Agent] 策略: {config.get('strategy', 'N/A')} | 理由: {config.get('rationale', 'N/A')}")

            # 运行实验
            round_output = os.path.join(OUTPUT_DIR, "task1", f"round{round_num}")
            result = train_classification_improved(
                npz_path=CLS_DATA_PATH,
                config=config,
                output_dir=round_output,
                device=device,
                n_ensemble=n_ensemble,
                use_label_prop=config.get("label_prop_alpha", 0.3) > 0,
            )

            val_acc = result["val_accuracy"]

            # 记录实验
            entry = {
                "round": round_num,
                "config": config,
                "val_accuracy": val_acc,
                "rationale": config.get("rationale", ""),
                "strategy": config.get("strategy", ""),
            }
            self.memory.append(entry)

            traj = result["trajectory_entry"].copy()
            traj["round"] = round_num
            traj["rationale"] = config.get("rationale", "")
            traj["strategy"] = config.get("strategy", "")
            self.trajectory.append(traj)

            if val_acc > best_metric:
                best_metric = val_acc
                best_result = result
                best_round = round_num
                # 保存最佳预测
                import shutil
                shutil.copy2(
                    os.path.join(round_output, "A1.csv"),
                    os.path.join(OUTPUT_DIR, "A1.csv")
                )

            print(f"\n[总结] 第 {round_num} 轮: Val Acc = {val_acc:.4f} | 最佳 = {best_metric:.4f}")

        # 保存轨迹
        traj_path = os.path.join(OUTPUT_DIR, "trajectory_B1.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_type": "classification",
                "total_rounds": len(self.trajectory),
                "experiments": self.trajectory,
                "best_result": max(self.memory, key=lambda x: x["val_accuracy"]),
            }, f, indent=2, ensure_ascii=False)
        print(f"[Agent] 轨迹已保存: {traj_path}")

        # 提炼跨任务经验
        self.cross_task_exp.append({
            "task": "classification",
            "best_config": max(self.memory, key=lambda x: x["val_accuracy"])["config"],
            "best_val_acc": best_metric,
            "key_findings": "L2特征归一化+对称归一化+DropEdge+集成+标签传播 有效",
        })

        print(f"\n[Task1 完成] 最佳 Val Acc = {best_metric:.4f} (第 {best_round} 轮)")
        return best_metric

    def run_recommendation(self, device="cuda"):
        """运行推荐任务的 Agent 多轮实验"""
        print("\n" + "=" * 70)
        print("  Task2: 产品推荐 - Agent 自动化实验")
        print("=" * 70)

        # 清空分类任务的记忆（保留跨任务经验）
        self.memory = []
        self.trajectory = []

        system_prompt = """你是 AFAC2026 自动化实验 Agent，负责序列推荐任务的实验优化。

## 任务信息
- 数据: 50,000 用户, 2,156 物品
- 训练用户 40,000 (平均序列长度 46), 测试用户 10,000 (平均序列长度 6.25)
- 评测指标: NDCG@10
- 模型: GRU4Rec (序列推荐)
- 关键挑战: 测试用户序列极短, 需要模型对短序列鲁棒

## 可调参数
- embedding_dim: 32/64/128
- hidden_dim: 64/128/256
- num_layers: 1/2
- dropout: 0.1/0.2/0.3
- lr: 0.0005/0.001/0.005
- max_seq_len: 20/50/100
- batch_size: 128/256/512
- epochs: 30/50/80
- use_seq_aug: true/false (序列增强: 随机截断模拟短序列)

## 决策原则
- 第1轮: 稳健 baseline
- 第2轮: 根据反馈探索
- 第3轮: 精细调优"""

        best_metric = 0.0
        best_result = None

        for round_num in range(1, self.budget + 1):
            print(f"\n{'─'*60}")
            print(f"  推荐任务 - 第 {round_num}/{self.budget} 轮")
            print(f"{'─'*60}")

            # LLM 决策
            history_str = json.dumps(self.memory, indent=2, ensure_ascii=False) if self.memory else "暂无历史"
            cross_str = json.dumps(self.cross_task_exp, indent=2, ensure_ascii=False) if self.cross_task_exp else "暂无"

            user_prompt = f"""## 当前状态
- 第 {round_num} 轮, 剩余预算 {self.budget - round_num + 1}
- 已完成实验: {len(self.memory)} 次

## 历史实验
{history_str}

## 跨任务经验 (来自分类任务)
{cross_str}

请分析历史反馈，输出下一轮实验配置 JSON:
```json
{{
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
    "loss_type": "ce",
    "use_seq_aug": true,
    "strategy": "exploration/exploitation",
    "rationale": "决策理由"
}}
```"""

            response = self._llm_chat(system_prompt, user_prompt)
            config = self._extract_json(response)

            if not config:
                config = {
                    "embedding_dim": 64, "hidden_dim": 128, "num_layers": 1,
                    "dropout": 0.2, "lr": 0.001, "weight_decay": 0,
                    "epochs": 50, "max_seq_len": 50, "batch_size": 256,
                    "patience": 7, "loss_type": "ce", "use_seq_aug": True,
                    "strategy": "default", "rationale": "默认配置"
                }

            config.setdefault("model_type", "gru4rec")
            config.setdefault("embedding_dim", 64)
            config.setdefault("hidden_dim", 128)
            config.setdefault("num_layers", 1)
            config.setdefault("dropout", 0.2)
            config.setdefault("lr", 0.001)
            config.setdefault("weight_decay", 0)
            config.setdefault("epochs", 50)
            config.setdefault("max_seq_len", 50)
            config.setdefault("batch_size", 256)
            config.setdefault("patience", 7)
            config.setdefault("loss_type", "ce")
            config.setdefault("use_seq_aug", True)
            config.setdefault("popularity_weight", 0.0)
            config.setdefault("remove_seen", False)

            print(f"[Agent] 配置: embed={config['embedding_dim']}, hidden={config['hidden_dim']}, "
                  f"lr={config['lr']}, seq_aug={config.get('use_seq_aug', True)}")
            print(f"[Agent] 策略: {config.get('strategy', 'N/A')} | 理由: {config.get('rationale', 'N/A')}")

            # 运行实验
            round_output = os.path.join(OUTPUT_DIR, "task2", f"round{round_num}")
            result = train_recommendation_improved(
                data_dir=REC_DATA_DIR,
                config=config,
                output_dir=round_output,
                device=device,
            )

            val_ndcg = result["val_ndcg"]

            # 记录实验
            entry = {
                "round": round_num,
                "config": config,
                "val_ndcg": val_ndcg,
                "rationale": config.get("rationale", ""),
                "strategy": config.get("strategy", ""),
            }
            self.memory.append(entry)

            traj = result["trajectory_entry"].copy()
            traj["round"] = round_num
            traj["rationale"] = config.get("rationale", "")
            traj["strategy"] = config.get("strategy", "")
            self.trajectory.append(traj)

            if val_ndcg > best_metric:
                best_metric = val_ndcg
                best_result = result
                best_round = round_num
                import shutil
                shutil.copy2(
                    os.path.join(round_output, "A2.csv"),
                    os.path.join(OUTPUT_DIR, "A2.csv")
                )

            print(f"\n[总结] 第 {round_num} 轮: Val NDCG@10 = {val_ndcg:.4f} | 最佳 = {best_metric:.4f}")

        # 保存轨迹
        traj_path = os.path.join(OUTPUT_DIR, "trajectory_B2.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_type": "recommendation",
                "total_rounds": len(self.trajectory),
                "experiments": self.trajectory,
                "best_result": max(self.memory, key=lambda x: x["val_ndcg"]),
            }, f, indent=2, ensure_ascii=False)
        print(f"[Agent] 轨迹已保存: {traj_path}")

        print(f"\n[Task2 完成] 最佳 Val NDCG@10 = {best_metric:.4f} (第 {best_round} 轮)")
        return best_metric


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 Agent 自动化实验")
    parser.add_argument("--budget", type=int, default=3, help="每任务实验轮次")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_ensemble", type=int, default=10, help="分类集成数量")
    parser.add_argument("--task", type=str, default="both", choices=["both", "cls", "rec"])
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 赛题三: Agent 自动化实验系统")
    print("  LLM 决策 + 多轮迭代 + 实验记忆 + 轨迹日志")
    print("=" * 70)
    print(f"  LLM 模型: {LLM_MODEL}")
    print(f"  实验预算: 每任务 {args.budget} 轮")
    print(f"  分类集成: {args.n_ensemble} 个模型")

    agent = AFACAgent(api_key=DEFAULT_API_KEY, budget=args.budget)
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
    print(f"\n  输出文件:")
    print(f"    A1.csv: {os.path.join(OUTPUT_DIR, 'A1.csv')}")
    print(f"    A2.csv: {os.path.join(OUTPUT_DIR, 'A2.csv')}")
    print(f"    trajectory_B1.json: {os.path.join(OUTPUT_DIR, 'trajectory_B1.json')}")
    print(f"    trajectory_B2.json: {os.path.join(OUTPUT_DIR, 'trajectory_B2.json')}")


if __name__ == "__main__":
    main()
