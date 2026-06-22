import json
from pathlib import Path
import os

corpus_path = Path("japanese-daily-dialogue/data/topic1.json")

with corpus_path.open("r", encoding="utf-8") as f:
    data = json.load(f)


"""for i in range(10):
    for j in range(10):
        print(data[i]["utterances"][j]["utterance"])
"""

for dialogue in data[:10]:
    print(dialogue["dialogue_id"])
    for utterance in dialogue["utterances"]:
        print(utterance["speaker"], utterance["utterance"])
        """
        print(data[i])
        print(data[i]["utterances"])
        """
