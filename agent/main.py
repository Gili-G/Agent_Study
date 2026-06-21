import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List,Dict

# 加载.env文件
load_dotenv()

# 创建一个LLM客户端类
class  LLM:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, message: List[Dict[str, str]], temperature: float = 0):

        print(f"正在使用{self.model}大模型")
        try:
            response = self.client.chat.completions.create(
                messages=message,
                model=self.model,
                temperature=temperature,
                stream=True
            )

            content = []
            for chunk in response:
                

