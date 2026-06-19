from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_path = Path("prompts/labelprompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")

corpus_path = Path("japanese-daily-dialogue/data/topic1.json")
data = json.loads(corpus_path.read_text(encoding="utf-8"))


def judge_self_disclosure(text):
    """1発話を自己開示判定する
    戻り値：0 または 1
    """
    response = client.responses.create(
        model="gpt-5-mini", instructions=instructions, input=json.dumps()
    )

    result = response.output_text.strip()

    if result == "1":
        return 1
    else:
        return 0

# ラベルを付与する関数
def label_dataset(utterances):
    
