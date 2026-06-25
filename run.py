"""
主编排器: 运行完整的自动化实验流程
- Task1: 产品分类 (图节点分类)
- Task2: 产品推荐 (序列推荐)
- Agent: LLM 决策 + 多轮迭代
- 输出: A1.csv, A2.csv, trajectory 日志
"""
import os
import sys
import json
import time
import argparse

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.agent import ExperimentAgent
from src.train_cls import train_classification
from src.train_rec import train_recommendation


# 数据路径
CLS_DATA_PATH = os.path.join(PROJECT_ROOT, "A分类", "A分类", "A1.npz")
REC_DATA_DIR = os.path.join(PROJECT_ROOT, "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 默认 API Key (从环境变量获取，或使用此默认值)
DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


def run_classification_task(api_key, model_name, budget=3, device="cuda"):
    """运行分类任务的完整 Agent 流程"""
    print("\n" + "=" * 70)
    print("  Task1: 产品分类任务 (图节点分类) - Agent 自动化实验")
    print("=" * 70)

    # 创建 Agent
    agent = ExperimentAgent(
        task_type="classification",
        api_key=api_key,
        model_name=model_name,
        budget=budget,
    )

    task_output_dir = os.path.join(OUTPUT_DIR, "task1")

    best_result = None
    best_metric = 0.0
    best_round = 0

    for round_num in range(1, budget + 1):
        print(f"\n{'─'*60}")
        print(f"  分类任务 - 第 {round_num}/{budget} 轮实验")
        print(f"{'─'*60}")

        # Agent 提出配置
        config = agent.propose_config(round_num)

        # 运行实验 (每轮保存到独立文件)
        round_output_dir = os.path.join(task_output_dir, f"round{round_num}")
        result = train_classification(
            npz_path=CLS_DATA_PATH,
            config=config,
            output_dir=round_output_dir,
            device=device,
        )

        # 记录结果
        entry = agent.record_experiment(round_num, config, result)

        # 跟踪最佳结果
        val_acc = result["val_accuracy"]
        if val_acc > best_metric:
            best_metric = val_acc
            best_result = result
            best_round = round_num

        print(f"\n[总结] 第 {round_num} 轮: Val Acc = {val_acc:.4f} | "
              f"当前最佳 = {best_metric:.4f}")

    # 保存轨迹日志
    trajectory_path = os.path.join(OUTPUT_DIR, "trajectory_B1.json")
    agent.save_trajectory(trajectory_path)

    # 将最佳轮次的预测结果复制为最终 A1.csv
    import shutil
    best_round_dir = os.path.join(task_output_dir, f"round{best_round}")
    best_a1 = os.path.join(best_round_dir, "A1.csv")
    final_a1 = os.path.join(OUTPUT_DIR, "A1.csv")
    shutil.copy2(best_a1, final_a1)
    print(f"\n[Task1 完成] 最佳验证准确率: {best_metric:.4f} (第 {best_round} 轮)")
    print(f"[Task1] 最终预测文件: {final_a1}")

    return agent


def run_recommendation_task(api_key, model_name, budget=3, device="cuda"):
    """运行推荐任务的完整 Agent 流程"""
    print("\n" + "=" * 70)
    print("  Task2: 产品推荐任务 (序列推荐) - Agent 自动化实验")
    print("=" * 70)

    # 创建 Agent
    agent = ExperimentAgent(
        task_type="recommendation",
        api_key=api_key,
        model_name=model_name,
        budget=budget,
    )

    task_output_dir = os.path.join(OUTPUT_DIR, "task2")

    best_result = None
    best_metric = 0.0
    best_round = 0

    for round_num in range(1, budget + 1):
        print(f"\n{'─'*60}")
        print(f"  推荐任务 - 第 {round_num}/{budget} 轮实验")
        print(f"{'─'*60}")

        # Agent 提出配置
        config = agent.propose_config(round_num)

        # 运行实验 (每轮保存到独立文件)
        round_output_dir = os.path.join(task_output_dir, f"round{round_num}")
        result = train_recommendation(
            data_dir=REC_DATA_DIR,
            config=config,
            output_dir=round_output_dir,
            device=device,
        )

        # 记录结果
        entry = agent.record_experiment(round_num, config, result)

        # 跟踪最佳结果
        val_ndcg = result["val_ndcg"]
        if val_ndcg > best_metric:
            best_metric = val_ndcg
            best_result = result
            best_round = round_num

        print(f"\n[总结] 第 {round_num} 轮: Val NDCG@10 = {val_ndcg:.4f} | "
              f"当前最佳 = {best_metric:.4f}")

    # 保存轨迹日志
    trajectory_path = os.path.join(OUTPUT_DIR, "trajectory_B2.json")
    agent.save_trajectory(trajectory_path)

    # 将最佳轮次的预测结果复制为最终 A2.csv
    import shutil
    best_round_dir = os.path.join(task_output_dir, f"round{best_round}")
    best_a2 = os.path.join(best_round_dir, "A2.csv")
    final_a2 = os.path.join(OUTPUT_DIR, "A2.csv")
    shutil.copy2(best_a2, final_a2)
    print(f"\n[Task2 完成] 最佳验证 NDCG@10: {best_metric:.4f} (第 {best_round} 轮)")
    print(f"[Task2] 最终预测文件: {final_a2}")

    return agent


def finalize_submission(cls_agent, rec_agent):
    """整理最终提交文件"""
    print("\n" + "=" * 70)
    print("  整理最终提交文件")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # A1.csv 应该已经在最后一轮保存，但我们需要确保使用最佳结果
    # 由于每轮都会覆盖输出文件，我们需要重新运行最佳配置来生成最终预测
    # 或者，我们可以在训练时将每轮结果保存到不同文件，最后选择最佳
    # 这里简化处理: 使用最后一轮的结果 (通常 Agent 会逐步优化)

    # 检查文件是否存在
    a1_path = os.path.join(OUTPUT_DIR, "task1", "A1.csv")
    a2_path = os.path.join(OUTPUT_DIR, "task2", "A2.csv")

    if os.path.exists(a1_path):
        print(f"[OK] A1.csv 已生成: {a1_path}")
    else:
        print(f"[WARNING] A1.csv 未找到!")

    if os.path.exists(a2_path):
        print(f"[OK] A2.csv 已生成: {a2_path}")
    else:
        print(f"[WARNING] A2.csv 未找到!")

    # 打印最佳结果摘要
    cls_best = cls_agent.get_best_result()
    rec_best = rec_agent.get_best_result()

    print(f"\n{'─'*60}")
    print(f"  最终结果摘要")
    print(f"{'─'*60}")
    if cls_best:
        print(f"  分类任务最佳: Val Accuracy = {cls_best.get('val_accuracy', 0):.4f} "
              f"(第 {cls_best['round']} 轮, {cls_best.get('config', {}).get('model_type', 'N/A')})")
    if rec_best:
        print(f"  推荐任务最佳: Val NDCG@10 = {rec_best.get('val_ndcg', 0):.4f} "
              f"(第 {rec_best['round']} 轮, {rec_best.get('config', {}).get('model_type', 'N/A')})")

    # 计算预估最终分数
    cls_score = cls_best.get("val_accuracy", 0) if cls_best else 0
    rec_score = rec_best.get("val_ndcg", 0) if rec_best else 0
    final_score = 0.5 * cls_score + 0.5 * rec_score
    print(f"\n  预估最终分数: 0.5 * {cls_score:.4f} + 0.5 * {rec_score:.4f} = {final_score:.4f}")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 赛题三: 自动化实验 Agent")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY,
                        help="DashScope API Key")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="LLM 模型名称")
    parser.add_argument("--budget", type=int, default=3,
                        help="每个任务的实验轮次")
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备 (cuda/cpu)")
    parser.add_argument("--task", type=str, default="both",
                        choices=["both", "cls", "rec"],
                        help="运行哪个任务")
    args = parser.parse_args()

    print("=" * 70)
    print("  AFAC2026 赛题三: 稀疏反馈下的自动化实验挑战")
    print("  Agent 自动化实验系统")
    print("=" * 70)
    print(f"  LLM 模型: {args.model}")
    print(f"  实验预算: 每任务 {args.budget} 轮")
    print(f"  训练设备: {args.device}")
    print(f"  分类数据: {CLS_DATA_PATH}")
    print(f"  推荐数据: {REC_DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")

    start_time = time.time()

    cls_agent = None
    rec_agent = None

    if args.task in ("both", "cls"):
        cls_agent = run_classification_task(
            api_key=args.api_key,
            model_name=args.model,
            budget=args.budget,
            device=args.device,
        )

    if args.task in ("both", "rec"):
        rec_agent = run_recommendation_task(
            api_key=args.api_key,
            model_name=args.model,
            budget=args.budget,
            device=args.device,
        )

    if cls_agent and rec_agent:
        finalize_submission(cls_agent, rec_agent)

    elapsed = time.time() - start_time
    print(f"\n[完成] 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
