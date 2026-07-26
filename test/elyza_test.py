from llama_cpp import Llama
import os
from pathlib import Path
import json
import time
from huggingface_hub import try_to_load_from_cache

PROMPT_PATH = Path("../prompts/labelprompt.txt")
SAVE_PATH = Path("../outputs/elyza_label_output.json")
CORPUS_PATH = Path("../japanese-daily-dialogue/data/topic1.json")

N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "256"))
MODEL_REPO = "elyza/Llama-3-ELYZA-JP-8B-GGUF"
MODEL_FILENAME = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_N_TOKENS", "128"))

with CORPUS_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)


def label_check(text):

    input_data = [
        {
            "turn_num": item["turn_num"],
            "speaker": item["speaker"],
            "utterance": item["utterance"],
        }
        for item in data
    ]
    print(input_data)


def main():
    # 空のリストを用意
    all_labels = []
    for dialogue in data[:5]:
        label_num = dialogue["dialogue_id"]
        label_check(data)
        print(label_num)


if __name__ == "__main__":
    main()
