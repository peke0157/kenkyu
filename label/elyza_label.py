from llama_cpp import Llama
import os
from pathlib import Path
import json
import time

prompt_path = Path("../prompts/labelprompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")
save_path = Path("../outputs/elyza_label_output.json")
corpus_path = Path("../japanese-daily-dialogue/data/topic1.json")


with corpus_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

llm = Llama.from_pretrained(
    repo_id="elyza/Llama-3-ELYZA-JP-8B-GGUF",
    filename="Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    chat_format="llama-3",
    n_ctx=1024,
)


def judge_self_disclosure(text):
    """1発話を自己開示判定する
    戻り値：0 または 1
    """
    """topic_payload = {
        "dialogue": 
    }"""

    llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ]
    )

    result = llm.create_chat_completion.output_text.strip()
    repr(result)
    result_list = json.loads(result)

    return result_list


# ラベルを付与する関数
def label_dataset(utterances):

    label_list = []

    input_list = []

    for dialogue_data in utterances:
        input_data = dialogue_data["utterance"]
        print(input_data)
        input_list.append(input_data)
        input_api = "\n".join(input_list)

    label = judge_self_disclosure(input_api)

    label_list.append(label)

    length = len(label_list)
    for i in range(length):

        print(f"{len(utterances)}発話ラベル処理完了")
        time.sleep(1)

    return label_list


# 判定結果を保存する
def save_list(label_list):
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)


def main():
    # 空のリストを用意
    all_labels = []
    for dialogue in data[:5]:
        label_num = dialogue["dialogue_id"]
        print(label_num)
        all_labels.append(label_num)
        label_list = label_dataset(dialogue["utterances"])
        all_labels.extend(label_list)
        print(all_labels)

    save_list(all_labels)
    print("保存完了")


if __name__ == "__main__":
    main()
