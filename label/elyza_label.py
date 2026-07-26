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

# 設定値の間違いを確認する
if N_CTX < 1024:
    raise ValueError("LLAMA_N_CTXは1024以上にしてください。")
if MAX_OUTPUT_TOKENS >= N_CTX:
    raise ValueError("LLAMA_N_TOKENSはLLAMA_N_CTXよりも小さくしてください。")

instructions = PROMPT_PATH.read_text(encoding="utf-8")

with CORPUS_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)


llama_options = {
    "chat_format": "llama-3",
    "n_ctx": N_CTX,
    "n_batch": N_BATCH,
    "use_mmap": True,
    "verbose": False,
}

model_path = os.getenv("LLAMA_MODEL_PATH")
if model_path is None:
    cached_path = try_to_load_from_cache(MODEL_REPO, MODEL_FILENAME)
    model_path = cached_path if isinstance(cached_path, str) else None
if model_path is not None:
    llm = Llama(model_path=model_path, **llama_options)

else:
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        **llama_options,
    )


def judge_self_disclosure(text):
    """1発話を自己開示判定する
    戻り値：0 または 1
    """
    """topic_payload = {
        "dialogue": 
    }"""

    start_time = time.perf_counter()
    max_tokens = min(MAX_OUTPUT_TOKENS, 32 + 4 * len(text))
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )

    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "出力がトークン時上限で途切れました。"
            "LLAMA_MAX_TOKENSを少し増やしてください。"
        )

    result = choice["message"]["content"].strip()

    print("モデルの生出力:", repr(result))
    result_list = json.loads(result)

    print(result_list)

    fin_time = time.perf_counter() - start_time
    print(fin_time)

    return result_list


# ラベルを付与する関数
def label_dataset(utterances):

    label_list = []

    input_list = []

    for dialogue_data in utterances:
        input_data = dialogue_data["utterance"]
        print(input_data)
        input_list.append(input_data)
        input_elyza = "\n".join(input_list)

    label = judge_self_disclosure(input_elyza)

    label_list.append(label)

    length = len(label_list)
    for i in range(length):

        print(f"{len(utterances)}発話ラベル処理完了")
        time.sleep(1)

    return label_list


# 判定結果を保存する
def save_list(label_list):
    with SAVE_PATH.open("w", encoding="utf-8") as f:
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

        save_list(all_labels)
    print("保存完了")


if __name__ == "__main__":
    main()
