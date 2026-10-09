import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

raw_path = Path("raw_article.json")
data = json.loads(raw_path.read_text(encoding="utf-8"))

prompt_path = Path("prompts/seed_plan_prompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")

schema = {
    "type": "json_schema",
    "name": "seed_plan_output",
    "schema": {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "seed_sentence_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["seed_sentence_id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sentences"],
        "additionalProperties": False,
    },
    "strict": True,
}

article_payload = {
    "article_id": data["article_id"],
    "title": data["title"],
    "body": data["body"],
}

response = client.responses.create(
    model="gpt-5-mini",
    instructions=instructions,
    input=json.dumps(article_payload, ensure_ascii=False),
    text={"format": schema},
)

structured = json.loads(response.output_text)

output = {
    "article_id": data["article_id"],
    "seed_plan": {"sentences": structured["sentences"]},
}

Path("seed_plan.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("seed_plan.json を保存しました。")
