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


response = client.responses.create(
    model="gpt-5-mini",
    instructions=instructions
    input=json.dumps()
)
