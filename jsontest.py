import json
from pathlib import Path
import os

corpus_path = Path("japanese-daily-dialogue/data/topic1.json")

with corpus_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

for i in range(10):

    print(data[i]["dialogue_length"])
