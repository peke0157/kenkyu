from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

schema = {
    "type": "json_schema",
    "name": "simple_summary",
    "schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
    "strict": True,
}

response = client.responses.create(
    model="gpt-5-mini",
    instructions="1文で要約してください",
    input="新しい駅前広場にはバス停とタクシー乗り場が整備されています。",
    text={"format": schema},
)

print(response.output_text)
