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
    default=PROJECT_ROOT / "corpus" / "esconv-ja" / "ESconvja.json",
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
# 1回のモデル呼び出しで判定する発話数（長い対話はこの数ずつに分けて判定する）
CHUNK_SIZE = int(os.getenv("LLAMA_CHUNK_SIZE", "8"))

# 設定値の間違いを確認する
if N_CTX < 2048:
    raise ValueError("LLAMA_N_CTXは2048以上にしてください。")
if MAX_OUTPUT_TOKENS >= N_CTX:
    raise ValueError("LLAMA_N_TOKENSはLLAMA_N_CTXよりも小さくしてください。")
if CHUNK_SIZE < 1:
    raise ValueError("LLAMA_CHUNK_SIZEは1以上にしてください。")

prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

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
    with Path(path).open("r", encoding="utf-8") as p_f:
        defs = yaml.safe_load(p_f)
    if not isinstance(defs, dict) or "gate" not in defs or "attributes" not in defs:
        raise ValueError(f"{path} に gate と attributes が必要です。")
    return defs


# 1属性の定義をテキスト化
def render_item(item):
    lines = [
        f"## {item['name']} (キー: {item['key']})",
        f"定義: {str(item['definition']).strip()}",
        "値:",
    ]
    for value, desc in item.get("values", {}).items():
        lines.append(f'- "{value}": {desc}')
    examples = item.get("example") or []
    if examples:
        lines.append("例:")
        for ex in examples:
            lines.append(f'- 発話: {ex["utterance"]} → "{ex["label"]}"')
    return "\n".join(lines)


# プロンプトの {{GATE}} と {{DEFINITIONS}} に定義を埋め込む
def build_instructions(template, defs):
    text = template.replace("{{GATE}}", render_item(defs["gate"]))
    text = text.replace(
        "{{DEFINITIONS}}",
        "\n\n".join(render_item(item) for item in defs["attributes"]),
    )
    if "{{" in text:
        raise ValueError("プロンプトに置換されていないプレースホルダーが残っています。")
    return text


definitions = load_definition()
instructions = build_instructions(prompt_template, definitions)


def make_response_schema(utterance_count, defs):
    """出力をJSONとしてスキーマを定義（キーと値は definitions.yaml から作る）"""
    label_items = [defs["gate"]] + list(defs["attributes"])
    properties = {"turn_num": {"type": "integer", "minimum": 1}}
    for item in label_items:
        properties[item["key"]] = {
            "type": "string",
            "enum": [str(v) for v in item["values"].keys()],
        }
    properties["reason"] = {"type": "string"}
    return {
        "type": "array",
        "minItems": utterance_count,
        "maxItems": utterance_count,
        "items": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
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

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_json},
    ]

    # 入力のトークン数を数え、コンテキストに収まる範囲で出力上限を決める
    # （64 はチャットテンプレートの特殊トークン分の余裕）
    prompt_tokens = sum(
        len(llm.tokenize(m["content"].encode("utf-8"), add_bos=False))
        for m in messages
    ) + 64
    max_tokens = min(MAX_OUTPUT_TOKENS, N_CTX - prompt_tokens)
    if max_tokens < 50 * len(text):
        raise RuntimeError(
            f"コンテキストが足りません（入力 約{prompt_tokens} tokens / N_CTX {N_CTX}）。"
            "LLAMA_CHUNK_SIZEを小さくするか、LLAMA_N_CTXを大きくしてください。"
        )

    start_time = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        response_format={
            "type": "json_object",
            "schema": make_response_schema(len(text), definitions),
        },
        temperature=0,
        max_tokens=max_tokens,
    )

    choice = response["choices"][0]
    result = choice["message"]["content"].strip()
    print("モデルの生出力:", repr(result))

    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "出力がトークン上限で途切れました。"
            "LLAMA_N_TOKENSを増やすか、LLAMA_CHUNK_SIZEを小さくしてください。"
        )

    try:
        result_list = json.loads(result)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"モデルの出力をJSONとして読めませんでした: {e}") from e

    expected = [item["turn_num"] for item in text]
    got = [r.get("turn_num") for r in result_list]
    if got != expected:
        print(f"警告: turn_numが入力と一致しません（入力 {expected} / 出力 {got}）")

    fin_time = time.perf_counter() - start_time
    usage = response.get("usage", {})

    print(
        f"{fin_time: .2f}s"
        f"(入力: {usage.get('prompt_tokens', '?')} tokens, "
        f"出力: {usage.get('completion_tokens', '?')} tokens)"
    )

    return result_list


# 対話を CHUNK_SIZE 発話ずつに分ける
def label_chunk(dialog, size=CHUNK_SIZE):
    return [dialog[i : i + size] for i in range(0, len(dialog), size)]


# ラベルを付与する関数（1対話分の判定結果をまとめて返す）
def label_dataset(dialog):
    label_list = []
    for chunk in label_chunk(dialog):
        label_list.extend(judge_self_disclosure(chunk))
    return label_list


# 判定結果を保存する
def save_list(chunk_list):
    with SAVE_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunk_list, f, ensure_ascii=False, indent=4)


def main():
    # 空のリストを用意
    all_labels = []
    for dialogue in data[:1]:
        dialogue_id = dialogue["dialogue_id"]
        print(dialogue_id)
        label_list = label_dataset(dialogue["dialog"])
        all_labels.append({"dialogue_id": dialogue_id, "labels": label_list})

        save_list(all_labels)
    print("保存完了")


if __name__ == "__main__":
    main()
