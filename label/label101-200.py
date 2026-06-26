from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_path = Path("../prompts/labelprompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")
save_path = Path("../outputs/label_output_101-200.json")
corpus_path = Path("../japanese-daily-dialogue/data/topic1.json")


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
    repr(result)
    result_list = json.loads(result)
    
    return result_list

# ラベルを付与する関数
def label_dataset(utterances):

    label_list = []
    
    input_list = []
    
    for dialogue_data in utterances:
        input_data = dialogue_data["utterance"]
        input_list.append(input_data)
        input_api = "\n".join(input_list)

    label = judge_self_disclosure(input_api)
    
    label_list.append(label)
    
    length = len(label_list)
    for i in range(length):
            
            print(f"{len(utterances)} ラベル処理完了")
            time.sleep(1)

    return label_list


# 判定結果を保存する
def save_list(label_list):
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)


def main():
    # 空のリストを用意
    all_labels = []
    for dialogue in data[100:200]:
        label_num = dialogue["dialogue_id"]
        print(label_num)
        all_labels.append(label_num)
        label_list = label_dataset(dialogue["utterances"])
        all_labels.extend(label_list)

    save_list(all_labels)
    print("保存完了")


if __name__ == "__main__":
    main()
