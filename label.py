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
save_path = Path("outputs/label_output.json")
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
    result_list = json.loads(result)
    
    self_disclosure = result_list[0]["self_disclosure"]
    
    if self_disclosure == "1":
        return 1
    else:
        return 0


# ラベルを付与する関数
def label_dataset(utterances):

    label_list = []
    
    for i, utterances_data in enumerate(utterances):
        text = utterances_data["utterance"]

        label = judge_self_disclosure(text)

        label_list.append(
            {
                "turn_num": utterances_data["turn_num"],
                "speaker": utterances_data["speaker"],
                "utterance": text,
                "self_disclosure": label,
            }
        )

        print(f"{i+1} / {len(utterances)} 完了")
        time.sleep(1)

    return label_list


# 判定結果を保存する
def save_list(label_list):
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)


def main():
    # 空のリストを用意
    all_labels = []
    for dialogue in data[:10]:
        label_num = dialogue["dialogue_id"]
        print(type(label_num))
        all_labels.append(label_num)
        label_list = label_dataset(dialogue["utterances"])
        all_labels.extend(label_list)

    save_list(all_labels)
    print("保存完了")


if __name__ == "__main__":
    main()
