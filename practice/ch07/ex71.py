from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5-mini",
    instructions="長文を2文または3文に分けてください",
    input="NPBは先日の打者のスイング時のバットが球審の頭部に直撃したことから、球審にもヘルメットの着用を義務化しました。",
)
print(response.output_text)
