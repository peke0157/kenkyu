import json
from pathlib import Path
import os

corpus_path = Path("japanese-daily-dialogue/data/topic1.json")
data = json.loads(corpus_path.read_text(encoding="utf-8"))

print(len(data))
print(data[0])
print(data[:10])
