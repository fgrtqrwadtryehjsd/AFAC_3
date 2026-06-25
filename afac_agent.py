import json
import os
import re
import numpy as np
from openai import OpenAI

class SandboxEnvironment:
    """沙盒环境：代理评估配置并返回稀疏反馈"""
    def __init__(self, task_type):
        self.task_type = task_type
        self.best_score = 0.0
        
    def evaluate(self, config):
        # 实际代码这里可以调用 GNN 训练并提取结果，这里用模拟给出行之有效的打分。
        # 结合分类和推荐两种数据集的特性
        score = 0.5
        feedback = "模型执行完毕。"
        
        if self.task_type == "classification":
            # 分类任务: 节点数为 13752，特征维度为 767，类别数为 10
            model = config.get("model", "GCN")
            if model in ["GAT", "GraphSAGE"]:
                score += 0.15
            
            layers = config.get("layers", 2)
            if layers > 3:
                feedback += " 层数过多，图网络出现过平滑现象，同质化严重。"
                score -= 0.1
            else:
                feedback += " 模型正常收敛，但对于罕见特征节点的分类准确率仍需提升。"
                score += 0.05
                
        elif self.task_type == "recommendation":
            # 推荐任务: uid序列推荐，候选项约2156个
            model = config.get("model", "LightGCN")
            lr = config.get("lr", 0.01)
            if model in ["LightGCN", "PinSage"]:
                score += 0.2
            
            if lr > 0.05:
                feedback += " 学习率设置过高，模型损失震荡不收敛。"
                score -= 0.15
            else:
                feedback += " 模型损失平稳下降，已提取出序列及交叉推荐特征。"
                score += 0.05
                
        self.best_score = max(self.best_score, score)
        return score, feedback

class AFACAgent:
    def __init__(self, api_key, task_type="classification", budget=3):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = "qwen3.5-flash"
        self.budget = budget
        self.task_type = task_type
        self.memory = []
        self.env = SandboxEnvironment(task_type)
        
    def get_llm_response(self, system_prompt, user_msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return str(e)
            
    def _extract_json(self, response_text):
        try:
            match = re.search(r'`json\s*(.*?)\s*`', response_text, re.DOTALL)
            if match: return json.loads(match.group(1))
            
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0: return json.loads(response_text[start:end])
        except Exception:
            pass
        return None

    def propose_config(self):
        # 根据数据集类型给模型喂食不同的先验知识
        dataset_info = ""
        if self.task_type == "classification":
            dataset_info = "数据集A1情况: 节点共13752个，特征维度767，图的稀疏比较大，无向异构，10个类别。"
        else:
            dataset_info = "数据集A推荐情况: 5万用户群，序列推荐特性，候选item有2156个，用户、item具有隐藏特征字典。"
            
        prompt = f"""你是一个参与 AFAC2026 金融智能创新大赛的 AutoML 代理。
当前任务类型: {'产品分类(图节点分类)' if self.task_type == 'classification' else '序列推荐'}。
上下文信息: {dataset_info}。由于是稀疏监控场景，需要在探索和利用之间做出权衡。
剩余尝试次数: {self.budget}次。

历史实验反馈：
{json.dumps(self.memory, indent=2, ensure_ascii=False) if self.memory else '暂无历史实验，请给出一个初始稳定的Baseline配置。'}

请分析历史经验并生成下一步的模型参数。你必须在回复中包含以下格式的 JSON：
`json
{{
    "model": "选型(例如GCN/GAT/GraphSAGE 或基于推荐的 ItemCF/LightGCN 等)",
    "lr": 0.01,
    "layers": 2,
    "hidden_channels": 64,
    "strategy": "exploration" 或 "exploitation",
    "rationale": "用一句话解释修改了什么及为什么"
}}
` """
        response = self.get_llm_response(prompt, "请输出 JSON 格式的下一步预测实验参数。")
        config = self._extract_json(response)
        
        if not config:
            print("[警告] JSON 提取失败。返回默认设置。")
            config = {"model": "GraphSAGE" if self.task_type=="classification" else "LightGCN", "lr": 0.01, "layers": 2, "hidden_channels": 64, "strategy": "default", "rationale": "Fallback schema."}
            
        print(f"\\n[Agent 决策] -> Model: {config.get('model')} (LR: {config.get('lr')}, Layers: {config.get('layers')})")
        print(f"[决策理由] -> {config.get('rationale')}")
        return config

    def run(self):
        print(f"=== 开始自动实验循环 | 任务: {self.task_type} | 预算: {self.budget}次 ===")
        while self.budget > 0:
            print(f"\\n-------------------- 实验剩余预算: {self.budget} --------------------")
            config = self.propose_config()
            score, feedback = self.env.evaluate(config)
            print(f"[环境反馈] F1/NDCG 得分: {score:.4f} | 日志反馈: {feedback}")
            
            self.memory.append({
                "round": len(self.memory) + 1,
                "config": config,
                "score": score,
                "feedback": feedback
            })
            self.budget -= 1
            
        print("\\n=== 资源耗尽，实验结束。总结历史结果：===")
        for m in self.memory:
            print(f"  Round {m['round']}: Score {m['score']:.4f} | Model {m['config'].get('model')}")
            
        best = max(self.memory, key=lambda x: x['score'])
        print(f"\\n最优配置为 Round {best['round']}，分数为 {best['score']:.4f}")

if __name__ == '__main__':
    API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API KEY
    # 执行分类任务验证
    agent_cls = AFACAgent(api_key=API_KEY, task_type="classification", budget=3)
    agent_cls.run()

    # 执行推荐任务验证
    print("\\n\\n=======================================================")
    agent_rec = AFACAgent(api_key=API_KEY, task_type="recommendation", budget=2)
    agent_rec.run()
