import json
from pathlib import Path
import os
import ast

corpus_path = Path("japanese-daily-dialogue/data/topic1.json")

with corpus_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
    
data_list = [item['dialogue_id'] for item in data]
print(data_list)


"""for i in range(10):
    for j in range(10):
        print(data[i]["utterances"][j]["utterance"])
"""

for dialogue in data[:10]:
    for utterance in dialogue["utterances"]:
        
        """
        print(data[i])
        print(data[i]["utterances"])
        """
