import pandas as pd
from pathlib import Path
import json

corpus_path = Path("../japanese-daily-dialogue/data/topic1.json")
with corpus_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
    
# utteranceをリストにする関数
def list_dataset(utterances):
    
    label = []
    for dialogue_data in utterances:
        utterance_data = dialogue_data["utterance"]
        label.extend(utterance_data)
    return label
    
# リストをcsvにする関数
def save_csv(label):
    label.to_csv("../outputs/label.csv", encoding="utf-8")

def main():
    all_list = []
    for dialogue in data[:200]:
        label_list = list_dataset(dialogue["utterances"])
        all_list.extend(label_list)
"""
df = pd.read_json("../outputs/label_output.json", encoding="utf-8")
df_1 = pd.read_json("../outputs/label_output_101-200.json", encoding="utf-8")

df.to_csv("../outputs/label_output.csv", encoding="utf-8")
df_1.to_csv("../outputs/label_output_101-200.csv", encoding="utf-8")
"""