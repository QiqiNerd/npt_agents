from dotenv import load_dotenv
load_dotenv()  # 默认读取当前工作目录的 .env
from openai import OpenAI
import os

# 读取环境变量中的 API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    # 调用模型列表 API 测试 key 是否可用
    models = client.models.list()
    print("API key 有效，可以使用！")
    print("可用模型示例:", [m.id for m in models.data[:5]])
except Exception as e:
    print("API key 测试失败：", e)