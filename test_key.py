from dotenv import load_dotenv
load_dotenv()  # By default, it reads the `.env` file in the current working directory.
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    models = client.models.list()
    print("API key good")
    print("Examples of available models:", [m.id for m in models.data[:5]])
except Exception as e:
    print("API key failed: ", e)