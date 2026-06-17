import os
from openai import OpenAI

class AFACAgent:
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = "qwen-max"  # 用户要求 qwen3.5-flash，考虑到兼容性通常阿里云对应的是 qwen-max/plus 等，若使用特定版本请确保名称正确
        # 注意：Qwen 3.5 目前在阿里云官方 API 名称通常为 qwen-max-2025-01-25 或 qwen-plus 等。
        # 按照用户描述 "qwen3.5-flash"，这里先设定一个占位，建议验证具体 API Model ID
        self.model_name = "qwen3.5-flash" 
    
    def get_response(self, system_prompt, user_input):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    API_KEY = "sk-ws-H.RPMYXLE.IVuv.MEUCIHpKgKle8C148rMkekleMZSeUM1r7LtCI8NUSuOw2vxAAiEAoFswwUheWNfyKg0zhbDD5uSI5EwLINkoWjVdQslwaAQ"
    agent = AFACAgent(api_key=API_KEY)
    
    # 简单的测试
    sys_msg = "你是一个金融竞赛专家，负责指导 AFAC2026 自动化实验挑战赛。"
    user_msg = "请简要分析一下这个赛赛题的核心难点。"
    
    print("Agent Response:")
    print(agent.get_response(sys_msg, user_msg))
