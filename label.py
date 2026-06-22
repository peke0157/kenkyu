from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_path = Path("prompts/labelprompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")
save_path = Path("outputs")
corpus_path = Path("japanese-daily-dialogue/data/topic1.json")


with corpus_path.open("r", encoding="utf-8") as f:
    data = json.load(f)


def judge_self_disclosure(text):
    """1発話を自己開示判定する
    戻り値：0 または 1
    """
    """topic_payload = {
        "dialogue": 
    }"""

    response = client.responses.create(
        model="gpt-5-mini", instructions=instructions, input=text
    )

    result = response.output_text.strip()

    if result == "1":
        return 1
    else:
        return 0


# ラベルを付与する関数
def label_dataset(utterances):

    label_list = []

    for i, text in enumerate(utterances):

        label = judge_self_disclosure(text)

        label_list.append({"text": text, "label": label})

        print(f"{i+1} / {len(utterances)} 完了")
        time.sleep(1)

    return label_list


# 判定結果を保存する
def save_list(label_list):
    with save_path.open("label_output.json", "w", encoding="utf-8") as f:
        json.dump(label_list, ensure_ascii=False, indent=4),


def main():
    for dialogue in data[:10]:
        for utterance in dialogue["utterances"]:
            judge_self_disclosure(utterance["utterance"])

    label_list = label_dataset(dialogue)

    save_list(label_list)
    print("保存完了")


if __name__ == "__main__":
    main()
