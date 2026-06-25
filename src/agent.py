"""
Agent 模块: 基于 LLM 的自动化实验决策智能体
- 使用 Qwen API 进行实验配置决策
- 维护实验记忆
- 多轮迭代优化
- 生成 trajectory 日志
"""
import os
import re
import json
import time
from openai import OpenAI


class ExperimentAgent:
    """自动化实验 Agent

    流程:
    1. 读取数据描述和历史实验结果
    2. LLM 分析反馈信号, 提出下一轮实验配置
    3. 运行实验, 收集验证指标
    4. 迭代优化, 直至预算耗尽
    """

    def __init__(self, task_type, api_key=None, model_name="qwen-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 budget=5):
        self.task_type = task_type  # "classification" or "recommendation"
        self.budget = budget
        self.memory = []  # 实验记忆
        self.trajectory = []  # 完整轨迹日志

        # LLM 客户端
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def _get_system_prompt(self):
        """根据任务类型生成系统提示词"""
        if self.task_type == "classification":
            return """你是一个参与 AFAC2026 金融智能创新大赛的 AutoML Agent，专注于图节点分类任务。

## 任务描述
- 数据集: 金融产品图，包含产品节点特征与节点间连接关系
- A榜数据: 13,752 个节点, 767 维特征, 10 个类别
- 训练节点: 11,001, 测试节点: 2,751
- 图结构稀疏，部分节点连接关系较少 (稀疏监督场景)
- 评测指标: Accuracy (准确率)

## 可选模型
- GraphSAGE (sage): 均值聚合，对稀疏图鲁棒，推荐默认选择
- GCN (gcn): 谱域卷积，简单高效
- GAT (gat): 注意力机制，表达能力强但易过拟合

## 可调超参数
- hidden_dim: 隐藏层维度 (64, 128, 256)
- num_layers: 层数 (1, 2, 3) - 注意: 层数过多会过平滑
- dropout: Dropout率 (0.0, 0.1, 0.3, 0.5)
- lr: 学习率 (0.001, 0.005, 0.01)
- weight_decay: 权重衰减 (0, 5e-4, 1e-3)
- epochs: 训练轮数 (100-300)
- normalization: 归一化方式 ("sym" 对称归一化, "rw" 随机游走归一化, "none")
- patience: 早停耐心值 (20-50)
- feat_norm: 特征归一化 ("l2" L2行归一化, "standard" 标准化, "none" 不归一化)

## 决策原则
- 稀疏图场景优先选择 GraphSAGE
- 避免层数过多 (>3层) 导致过平滑
- 注意探索与利用的平衡
- 根据验证集反馈调整策略"""

        else:  # recommendation
            return """你是一个参与 AFAC2026 金融智能创新大赛的 AutoML Agent，专注于序列推荐任务。

## 任务描述
- 数据集: 金融场景序列推荐，用户-产品交互序列
- A榜数据: 50,000 用户, 2,156 个候选产品
- 训练用户: 40,000, 测试用户: 10,000
- 新用户交互历史较少 (稀疏监督场景)
- 评测指标: NDCG@10

## 可选模型
- GRU4Rec (gru4rec): GRU序列建模，参数少训练快，推荐默认选择
- SASRec (sasrec): Transformer，表达能力强但需要更多训练时间

## 可调超参数
- embedding_dim: 嵌入维度 (32, 64, 128)
- hidden_dim: 隐藏层维度 (64, 128, 256)
- num_layers: 层数 (1, 2)
- dropout: Dropout率 (0.0, 0.1, 0.2, 0.3)
- lr: 学习率 (0.001, 0.005, 0.01)
- weight_decay: 权重衰减 (0, 1e-4, 1e-3)
- epochs: 训练轮数 (20-100)
- max_seq_len: 最大序列长度 (20, 50, 100)
- batch_size: 批大小 (128, 256, 512)
- patience: 早停耐心值 (3-10)

## 决策原则
- 新用户历史少，不宜用过长序列
- 注意学习率不要过高导致不收敛
- 平衡模型容量与训练速度
- 根据验证集 NDCG 调整策略"""

    def _build_user_prompt(self, round_num):
        """构建用户提示词，包含历史实验记忆"""
        dataset_info = ""
        if self.task_type == "classification":
            dataset_info = """## 数据集信息
- 节点数: 13,752, 特征维度: 767, 类别数: 10
- 训练节点: 11,001, 测试节点: 2,751
- 图结构稀疏, 部分节点连接关系少
- 目标: 对连接稀疏的节点进行准确分类"""
        else:
            dataset_info = """## 数据集信息
- 用户数: 50,000, 候选产品数: 2,156
- 训练用户: 40,000, 测试用户: 10,000
- 新用户交互历史较少
- 目标: 为测试用户预测按置信度排序的 Top-10 产品列表"""

        history_str = "暂无历史实验，请给出一个稳定的初始 Baseline 配置。"
        if self.memory:
            history_str = json.dumps(self.memory, indent=2, ensure_ascii=False)

        remaining = self.budget - len(self.memory)

        prompt = f"""{dataset_info}

## 当前状态
- 第 {round_num} 轮实验
- 剩余预算: {remaining} 次
- 已完成实验: {len(self.memory)} 次

## 历史实验记录
{history_str}

## 任务
请分析历史实验反馈，决定下一轮实验配置。你需要在探索和利用之间做出权衡。

请输出一个 JSON 配置对象，包含以下字段:

对于分类任务 (classification):
```json
{{
    "model_type": "sage",
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 0.0005,
    "epochs": 300,
    "normalization": "sym",
    "patience": 50,
    "feat_norm": "l2",
    "strategy": "exploration 或 exploitation",
    "rationale": "一句话解释你的决策逻辑"
}}
```

对于推荐任务 (recommendation):
```json
{{
    "model_type": "gru4rec",
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 0,
    "epochs": 30,
    "max_seq_len": 50,
    "batch_size": 256,
    "patience": 5,
    "strategy": "exploration 或 exploitation",
    "rationale": "一句话解释你的决策逻辑"
}}
```

请直接输出 JSON 配置，不要输出其他内容。"""

        return prompt

    def _extract_json(self, response_text):
        """从 LLM 响应中提取 JSON 配置"""
        # 尝试从代码块中提取
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接提取 JSON
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

        return None

    def _get_default_config(self, round_num):
        """获取默认配置 (LLM 调用失败时的回退方案)"""
        if self.task_type == "classification":
            if round_num == 1:
                return {
                    "model_type": "sage", "hidden_dim": 256, "num_layers": 2,
                    "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
                    "epochs": 300, "normalization": "sym", "patience": 50,
                    "feat_norm": "l2",
                    "strategy": "baseline", "rationale": "默认 GraphSAGE baseline: L2特征归一化+对称归一化"
                }
            elif round_num == 2:
                return {
                    "model_type": "sage", "hidden_dim": 256, "num_layers": 3,
                    "dropout": 0.3, "lr": 0.005, "weight_decay": 5e-4,
                    "epochs": 300, "normalization": "sym", "patience": 50,
                    "feat_norm": "l2",
                    "strategy": "exploration", "rationale": "增加层数扩大感受野，降低学习率精细调优"
                }
            else:
                return {
                    "model_type": "sage", "hidden_dim": 128, "num_layers": 2,
                    "dropout": 0.5, "lr": 0.01, "weight_decay": 1e-3,
                    "epochs": 300, "normalization": "rw", "patience": 50,
                    "feat_norm": "l2",
                    "strategy": "exploitation", "rationale": "尝试随机游走归一化，增加正则化"
                }
        else:
            if round_num == 1:
                return {
                    "model_type": "gru4rec", "embedding_dim": 64, "hidden_dim": 128,
                    "num_layers": 1, "dropout": 0.2, "lr": 0.001, "weight_decay": 0,
                    "epochs": 30, "max_seq_len": 50, "batch_size": 256, "patience": 5,
                    "strategy": "baseline", "rationale": "默认 GRU4Rec baseline 配置"
                }
            elif round_num == 2:
                return {
                    "model_type": "gru4rec", "embedding_dim": 128, "hidden_dim": 256,
                    "num_layers": 1, "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-4,
                    "epochs": 50, "max_seq_len": 50, "batch_size": 256, "patience": 5,
                    "strategy": "exploration", "rationale": "增大嵌入和隐藏层维度，提升模型容量"
                }
            else:
                return {
                    "model_type": "sasrec", "embedding_dim": 64, "hidden_dim": 128,
                    "num_layers": 2, "dropout": 0.2, "lr": 0.001, "weight_decay": 0,
                    "epochs": 50, "max_seq_len": 50, "batch_size": 256, "patience": 5,
                    "strategy": "exploitation", "rationale": "尝试 SASRec Transformer 模型"
                }

    def propose_config(self, round_num):
        """使用 LLM 提出下一轮实验配置"""
        print(f"\n[Agent] 第 {round_num} 轮: 正在请求 LLM 生成实验配置...")

        try:
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(round_num)

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            response = completion.choices[0].message.content
            config = self._extract_json(response)

            if config:
                # 确保所有必需字段都存在
                if self.task_type == "classification":
                    config.setdefault("model_type", "sage")
                    config.setdefault("hidden_dim", 256)
                    config.setdefault("num_layers", 2)
                    config.setdefault("dropout", 0.5)
                    config.setdefault("lr", 0.01)
                    config.setdefault("weight_decay", 5e-4)
                    config.setdefault("epochs", 300)
                    config.setdefault("normalization", "sym")
                    config.setdefault("patience", 50)
                    config.setdefault("feat_norm", "l2")
                else:
                    config.setdefault("model_type", "gru4rec")
                    config.setdefault("embedding_dim", 64)
                    config.setdefault("hidden_dim", 128)
                    config.setdefault("num_layers", 1)
                    config.setdefault("dropout", 0.2)
                    config.setdefault("lr", 0.001)
                    config.setdefault("weight_decay", 0)
                    config.setdefault("epochs", 30)
                    config.setdefault("max_seq_len", 50)
                    config.setdefault("batch_size", 256)
                    config.setdefault("patience", 5)

                config.setdefault("strategy", "exploration")
                config.setdefault("rationale", "LLM 生成配置")

                print(f"[Agent] LLM 决策: {config.get('model_type')} | "
                      f"策略: {config.get('strategy')} | "
                      f"理由: {config.get('rationale')}")
                return config

        except Exception as e:
            print(f"[Agent] LLM 调用失败: {e}")

        # 回退到默认配置
        default_config = self._get_default_config(round_num)
        print(f"[Agent] 使用默认配置: {default_config.get('model_type')} | "
              f"策略: {default_config.get('strategy')}")
        return default_config

    def record_experiment(self, round_num, config, result):
        """记录实验结果到记忆和轨迹"""
        # 提取关键指标
        if self.task_type == "classification":
            metric = result.get("val_accuracy", 0.0)
            metric_name = "val_accuracy"
        else:
            metric = result.get("val_ndcg", 0.0)
            metric_name = "val_ndcg"

        entry = {
            "round": round_num,
            "config": config,
            metric_name: metric,
            "feedback": result.get("trajectory_entry", {}).get("feedback", ""),
            "epochs_trained": result.get("trajectory_entry", {}).get("epochs_trained", 0),
            "elapsed_seconds": result.get("trajectory_entry", {}).get("elapsed_seconds", 0),
            "strategy": config.get("strategy", "unknown"),
            "rationale": config.get("rationale", ""),
        }

        self.memory.append(entry)

        # 完整轨迹 (包含训练历史)
        trajectory_entry = result.get("trajectory_entry", {}).copy()
        trajectory_entry["round"] = round_num
        trajectory_entry["config"] = config
        trajectory_entry["rationale"] = config.get("rationale", "")
        trajectory_entry["strategy"] = config.get("strategy", "")
        trajectory_entry["next_strategy_hint"] = self._generate_next_hint(round_num, metric)
        self.trajectory.append(trajectory_entry)

        return entry

    def _generate_next_hint(self, round_num, current_metric):
        """根据当前结果生成下一轮优化提示"""
        if round_num == 0 or len(self.memory) < 2:
            return "继续探索不同模型结构和超参数组合"

        prev_metric = self.memory[-2].get(
            "val_accuracy" if self.task_type == "classification" else "val_ndcg", 0.0
        )

        if current_metric > prev_metric:
            return f"当前方向有效 ({current_metric:.4f} > {prev_metric:.4f})，继续在此方向上精细调优"
        else:
            return f"当前方向效果不佳 ({current_metric:.4f} <= {prev_metric:.4f})，尝试切换到不同策略"

    def get_best_result(self):
        """获取最佳实验结果"""
        if not self.memory:
            return None

        metric_name = "val_accuracy" if self.task_type == "classification" else "val_ndcg"
        best = max(self.memory, key=lambda x: x.get(metric_name, 0.0))
        return best

    def save_trajectory(self, filepath):
        """保存轨迹日志为 JSON 文件"""
        output = {
            "task_type": self.task_type,
            "total_rounds": len(self.trajectory),
            "experiments": self.trajectory,
            "best_result": self.get_best_result(),
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[Agent] 轨迹日志已保存至 {filepath}")
