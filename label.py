from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
prompt_path = Path("")
instructions = prompt_path.read_text(encoding="utf-8")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
