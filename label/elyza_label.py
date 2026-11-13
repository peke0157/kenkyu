# このプログラムはESconvのコーパスを量子化版のllama-3-elyzaでラベル付けするためのプログラム
from llama_cpp import Llama
import os
from pathlib import Path
import json
import time
from huggingface_hub import try_to_load_from_cache
import yaml
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corpus",
    type=Path,
    default=PROJECT_ROOT
    / "corpus"
    / "Emotional-Support-Conversation"
    / "ESConvja.json",
)
args = parser.parse_args()
CORPUS_PATH = args.corpus.expanduser().resolve()

DEF_PATH = PROJECT_ROOT / "definitions.yaml"
PROMPT_PATH = PROJECT_ROOT / "prompts/self-disclosure.md"
SAVE_PATH = PROJECT_ROOT / "outputs/elyza_label_output_test.json"


N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))
MODEL_REPO = "elyza/Llama-3-ELYZA-JP-8B-GGUF"
MODEL_FILENAME = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_N_TOKENS", "2048"))

# 設定値の間違いを確認する
if N_CTX < 2048:
    raise ValueError("LLAMA_N_CTXは2048以上にしてください。")
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


def load_definition(path=DEF_PATH):
    with DEF_PATH.open("r", encoding="utf-8") as p_f:
        defs = yaml.safe_load(p_f)
    return defs


# 1属性の定義をテキスト化
def render_item(item):
    lines = [
        f"## {item['name']} (キー: {item['key']})",
        f"定義: {item['definition'].strip()}",
    ]


def make_response_schema(utterance_count):
    """出力をJSONとしてスキーマを定義"""
    return {
        "type": "array",
        "minItems": utterance_count,
        "maxItems": utterance_count,
        "items": {
            "type": "object",
            "properties": {
                "turn_num": {
                    "type": "integer",
                    "minimum": 1,
                },
                "self_disclosure": {
                    "type": "string",
                    "enum": ["0", "1"],
                },
                "actual_damage": {
                    "type": "string",
                    "enum": ["0", "1"],
                },
                "depth": {
                    "type": "string",
                    "enum": ["0", "1", "2", "3", "4"],
                },
                "introspection": {
                    "type": "string",
                    "enum": ["0", "1"],
                },
                "negative_aspects": {
                    "type": "string",
                    "enum": ["0", "1"],
                },
                "psychological_resistance": {
                    "type": "string",
                    "enum": ["0", "1"],
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": [
                "turn_num",
                "self_disclosure",
                "actual_damage",
                "depth",
                "introspection",
                "negative_aspects",
                "psychological_resistance",
                "string",
            ],
            "additionalProperties": False,
        },
    }


def judge_self_disclosure(text):

    input_data = [
        {
            "turn_num": item["turn_num"],
            "speaker": item["speaker"],
            "content": item["content"],
        }
        for item in text
    ]
    input_json = json.dumps(input_data, ensure_ascii=False)

    start_time = time.perf_counter()
    max_tokens = min(MAX_OUTPUT_TOKENS, 64 + 100 * len(text))
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_json},
        ],
        response_format={
            "type": "json_object",
            "schema": make_response_schema(len(text)),
        },
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

    fin_time = time.perf_counter() - start_time
    usage = response.get("usage", {})

    print(
        f"{fin_time: .2f}s"
        f"(入力: {usage.get('prompt_tokens', '?')} tokens, "
        f"出力: {usage.get('completion_tokens', '?')} tokens)"
    )

    return result_list


# ラベルを付与する関数
def label_dataset(content):

    label_list = []

    label = judge_self_disclosure(content)

    label_list.append(label)

    length = len(label_list)
    return label_list


def label_chunk(label_list):
    chunk_list = []


# 判定結果を保存する
def save_list(chunk_list):
    with SAVE_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunk_list, f, ensure_ascii=False, indent=4)


def main():
    # 空のリストを用意
    all_labels = []
    for dialogue in data[:1]:
        label_num = dialogue["dialogue_id"]
        print(label_num)
        all_labels.append(label_num)
        label_list = label_dataset(dialogue["content"])
        all_labels.extend(label_list)

        save_list(all_labels)
    print("保存完了")


if __name__ == "__main__":
    main()
