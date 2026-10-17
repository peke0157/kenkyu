from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5-mini",
    instructions="40字以内で要約してください",
    input="この大学では先月、Tキャンパスで大学祭を行い、総勢1万人が来校されました。",
)
print(response.output_text)
