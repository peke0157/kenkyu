import json
import os
import time
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "../prompts/labelprompt.txt"
SAVE_PATH = BASE_DIR / "../outputs/elyza_label_output.json"
CORPUS_PATH = BASE_DIR / "../japanese-daily-dialogue/data/topic1.json"

# 8BモデルをRAM 8 GiB程度のCPU環境で動かすため、必要以上に大きくしない。
# n_ctxは「入力 + 出力」の合計上限であり、大きくするとKVキャッシュも増える。
N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "256"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "128"))
DIALOGUE_LIMIT = int(os.getenv("DIALOGUE_LIMIT", "5"))
MODEL_REPO = "elyza/Llama-3-ELYZA-JP-8B-GGUF"
MODEL_FILENAME = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"

if N_CTX < 1024:
    raise ValueError("LLAMA_N_CTXは1024以上にしてください。")
if MAX_OUTPUT_TOKENS >= N_CTX:
    raise ValueError("LLAMA_MAX_TOKENSはLLAMA_N_CTXより小さくしてください。")

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

# キャッシュ済みならHubへの問い合わせを省き、オフラインでもすぐ読み込む。
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


def _response_schema(number_of_utterances):
    """短い出力だけを許可するJSON Schemaを作る。"""
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "minItems": number_of_utterances,
                "maxItems": number_of_utterances,
                "items": {"type": "string", "enum": ["0", "1"]},
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def judge_self_disclosure(utterances):
    """対話内の各発話を自己開示判定し、元の発話情報と結合して返す。"""
    start_time = time.perf_counter()

    # 発話全文をモデルに再生成させると遅くなるため、判定結果だけを要求する。
    input_data = [
        {
            "turn_num": item["turn_num"],
            "speaker": item["speaker"],
            "utterance": item["utterance"],
        }
        for item in utterances
    ]
    user_message = (
        "以下の対話を判定してください。\n"
        "発話の順番を変えず、判定値だけを labels 配列に入れた"
        'JSONオブジェクト（例: {"labels":["0","1"]}）を返してください。\n'
        f"{json.dumps(input_data, ensure_ascii=False)}"
    )

    # 0/1の配列だけなので短い上限で十分。異常時も生成し続けない。
    max_tokens = min(MAX_OUTPUT_TOKENS, 32 + 4 * len(utterances))
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_object",
            "schema": _response_schema(len(utterances)),
        },
        temperature=0.0,
        max_tokens=max_tokens,
    )

    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "出力がトークン上限で途切れました。"
            "LLAMA_MAX_TOKENSを少し増やしてください。"
        )

    raw_result = choice["message"]["content"].strip()
    parsed = json.loads(raw_result)
    predictions = parsed["labels"]
    if len(predictions) != len(utterances):
        raise ValueError("モデルの出力したラベル数が入力発話数と一致しません。")

    result = [
        {
            "turn_num": item["turn_num"],
            "speaker": item["speaker"],
            "utterance": item["utterance"],
            "self_disclosure": prediction,
        }
        for item, prediction in zip(utterances, predictions)
    ]

    elapsed = time.perf_counter() - start_time
    usage = response.get("usage", {})
    print("判定結果:", result)
    print(
        f"応答時間: {elapsed:.2f}秒 "
        f"(入力: {usage.get('prompt_tokens', '?')} tokens, "
        f"出力: {usage.get('completion_tokens', '?')} tokens)"
    )
    return result


def label_dataset(utterances):
    for item in utterances:
        print(f"{item['speaker']}: {item['utterance']}")

    labels = judge_self_disclosure(utterances)
    print(f"{len(utterances)}発話のラベル処理完了")
    return labels


def save_list(label_list):
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAVE_PATH.open("w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)


def main():
    all_labels = []
    for dialogue in data[:DIALOGUE_LIMIT]:
        dialogue_id = dialogue["dialogue_id"]
        print(f"\n対話ID: {dialogue_id}")
        all_labels.append(dialogue_id)
        all_labels.append(label_dataset(dialogue["utterances"]))

        # 長時間実行中に停止しても、それまでの結果が残るよう逐次保存する。
        save_list(all_labels)

    print(f"保存完了: {SAVE_PATH.resolve()}")


if __name__ == "__main__":
    main()
