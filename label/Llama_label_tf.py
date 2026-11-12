# Llama-3-ELYZA-JP-8BのTransformer版のプログラム（Silverで実行）
"""Llama-3-ELYZA-JP-8B（Transformers版）でESConvの発話に自己開示ラベルを付与する。

モデル: https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B （GGUF版・llama-cppは使わない）

silverなど別環境での実行を想定し、コーパス・プロンプト・出力先は
コマンドライン引数か環境変数で指定できるようにしている。

例:
    python Llama_label_tf.py --corpus /path/to/ESconvja.json --limit 3
    ESCONV_PATH=/path/to/ESconvja.json python Llama_label_tf.py
    python Llama_label_tf.py --dry-run   # モデルを読まずに入力変換とトークン数だけ確認
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

# 優先順位: コマンドライン引数 > 環境変数 > 既定値（このリポジトリ内の配置）
DEFAULT_CORPUS = os.getenv(
    "ESCONV_PATH",
    str(REPO_DIR / "corpus/Emotional-Support-Conversation/ESconvja.json"),
)
DEFAULT_PROMPT = os.getenv("LABEL_PROMPT_PATH", str(REPO_DIR / "prompts/self-disclosure.md"))
DEFAULT_OUTPUT = os.getenv("LABEL_OUTPUT_PATH", str(REPO_DIR / "outputs/elyza_tf_label_output.json"))
DEFAULT_MODEL = os.getenv("ELYZA_MODEL", "elyza/Llama-3-ELYZA-JP-8B")
MAX_OUTPUT_TOKENS = int(os.getenv("LLAMA_N_TOKENS", "4096"))
MODEL_MAX_CTX = 8192  # Llama 3 のコンテキスト長

LABEL_KEYS = (
    "self_disclosure",
    "actual_damage",
    "depth",
    "introspection",
    "negative_aspects",
    "psychological_resistance",
)


def make_response_schema(utterance_count):
    """出力をJSONとしてスキーマを定義（elyza_label.py と同一の形式）"""
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
                "speaker": {
                    "type": "string",
                },
                "utterance": {
                    "type": "string",
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
            },
            "required": [
                "turn_num",
                "speaker",
                "utterance",
                "self_disclosure",
                "actual_damage",
                "depth",
                "introspection",
                "negative_aspects",
                "psychological_resistance",
            ],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# コーパス読み込み（ESConv形式 → turn_num/speaker/utterance 形式）
# ---------------------------------------------------------------------------
def load_esconv(path):
    """ESConv(.json)を読み、[{dialogue_id, utterances:[...]}] に変換する。

    対応形式:
      - ESConv.json / ESconvja.json : [{..., "dialog": [{"speaker", "content", ...}]}]
      - dialogue_id 付きの派生ファイル : [{"dialogue_id", "dialog": [...]}]
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"コーパスが見つかりません: {path}\n"
            "--corpus または環境変数 ESCONV_PATH でESConvのパスを指定してください。"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("ESConvのJSONは対話の配列である必要があります。")

    dialogues = []
    for index, item in enumerate(raw, start=1):
        dialog = item.get("dialog")
        if not isinstance(dialog, list):
            raise ValueError(f"{index}件目の対話に dialog 配列がありません。")
        utterances = [
            {
                "turn_num": u.get("turn_num", turn),  # 整形済みESConvはturn_numを持つ
                "speaker": u["speaker"],
                "utterance": u["content"].strip(),
            }
            for turn, u in enumerate(dialog, start=1)
        ]
        dialogues.append(
            {"dialogue_id": item.get("dialogue_id", index), "utterances": utterances}
        )
    return dialogues


# ---------------------------------------------------------------------------
# 出力の検証（Transformersには文法制約付き生成が無いため、生成後にスキーマを検証する）
# ---------------------------------------------------------------------------
def extract_json_array(text):
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("出力にJSON配列が見つかりません。")
    return json.loads(text[start : end + 1])


def validate_result(result, utterances):
    """make_response_schema に沿っているかを確認し、発話本文は入力の原文に揃える。"""
    schema_items = make_response_schema(len(utterances))["items"]
    props = schema_items["properties"]
    required = set(schema_items["required"])

    if not isinstance(result, list) or len(result) != len(utterances):
        raise ValueError(
            f"出力の発話数({len(result) if isinstance(result, list) else '?'})"
            f"が入力({len(utterances)})と一致しません。"
        )
    for original, labeled in zip(utterances, result):
        if not isinstance(labeled, dict) or set(labeled) != required:
            raise ValueError(f"turn {original['turn_num']}: キーがスキーマと一致しません。")
        if labeled["turn_num"] != original["turn_num"]:
            raise ValueError(f"turn {original['turn_num']}: turn_numの順序が入力と一致しません。")
        if labeled["speaker"] != original["speaker"]:
            raise ValueError(f"turn {original['turn_num']}: speakerが入力と一致しません。")
        for key in LABEL_KEYS:
            value = labeled[key]
            if isinstance(value, int) and not isinstance(value, bool):
                value = str(value)  # 数値で返ってきた場合は文字列に揃える
            if value not in props[key]["enum"]:
                raise ValueError(f"turn {original['turn_num']}: {key}={labeled[key]!r} が範囲外です。")
            labeled[key] = value
        if labeled["utterance"] != original["utterance"]:
            # モデルが発話を書き換えた場合も、ラベルは使い原文を保存する
            print(f"  [warn] turn {original['turn_num']}: 発話本文が改変されたため原文に戻しました。")
            labeled["utterance"] = original["utterance"]
    return result


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"dtype": "auto"}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16
        try:
            import accelerate  # noqa: F401

            kwargs["device_map"] = "auto"
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if "device_map" not in kwargs and torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    print(f"モデル読み込み完了: {model_name} ({model.device}, {model.dtype})")
    return tokenizer, model


def build_inputs(tokenizer, instructions, utterances):
    input_json = json.dumps(utterances, ensure_ascii=False)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_json},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # JSON配列から書き始めさせる（出力の先頭に説明文が入るのを防ぐ）
    prompt += "["
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)


def judge_self_disclosure(utterances, tokenizer, model, instructions, max_retries):
    inputs = build_inputs(tokenizer, instructions, utterances).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = min(MAX_OUTPUT_TOKENS, 64 + 150 * len(utterances))
    if input_len + max_new_tokens > MODEL_MAX_CTX:
        raise RuntimeError(
            f"入力{input_len}+出力{max_new_tokens}トークンがコンテキスト長"
            f"{MODEL_MAX_CTX}を超えます。--chunk-size で分割してください。"
        )

    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    last_error = None
    for attempt in range(1, max_retries + 1):
        # 1回目は貪欲法（temperature=0相当）、再試行時は少しだけサンプリングする
        gen_kwargs = {"do_sample": False}
        if attempt > 1:
            gen_kwargs = {"do_sample": True, "temperature": 0.3, "top_p": 0.9}

        start_time = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )
        new_tokens = output[0][input_len:]
        result = "[" + tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - start_time
        print(f"  {elapsed:.2f}s (入力: {input_len} tokens, 出力: {len(new_tokens)} tokens)")

        if len(new_tokens) >= max_new_tokens:
            last_error = "出力がトークン上限で途切れました（LLAMA_N_TOKENSを増やしてください）。"
        else:
            try:
                return validate_result(extract_json_array(result), utterances)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = str(e)
        print(f"  [retry {attempt}/{max_retries}] {last_error}")
        print("  モデルの生出力:", repr(result[:500]))

    raise RuntimeError(f"有効なJSONが得られませんでした: {last_error}")


def label_dialogue(utterances, tokenizer, model, instructions, chunk_size, max_retries):
    """対話をchunk_size発話ごとに判定し、1つのラベル配列にまとめる（0なら分割しない）"""
    if chunk_size <= 0:
        chunks = [utterances]
    else:
        chunks = [utterances[i : i + chunk_size] for i in range(0, len(utterances), chunk_size)]
    labels = []
    for chunk in chunks:
        labels.extend(judge_self_disclosure(chunk, tokenizer, model, instructions, max_retries))
    return labels


def save_list(label_list, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="ESConvのJSON（env: ESCONV_PATH）")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="判定プロンプト（env: LABEL_PROMPT_PATH）")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="出力JSON（env: LABEL_OUTPUT_PATH）")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HFモデル名またはローカルパス（env: ELYZA_MODEL）")
    p.add_argument("--start", type=int, default=0, help="処理を始める対話の位置（0始まり）")
    p.add_argument("--limit", type=int, default=None, help="処理する対話数（省略時は全件）")
    p.add_argument("--chunk-size", type=int, default=0, help="1回の推論に渡す発話数（0で対話全体）")
    p.add_argument("--max-retries", type=int, default=3, help="JSON検証に失敗したときの試行回数")
    p.add_argument("--dry-run", action="store_true", help="モデルを読み込まず入力変換とトークン数だけ確認")
    return p.parse_args()


def main():
    args = parse_args()
    instructions = Path(args.prompt).expanduser().read_text(encoding="utf-8")
    dialogues = load_esconv(args.corpus)
    end = None if args.limit is None else args.start + args.limit
    targets = dialogues[args.start : end]
    print(f"コーパス: {args.corpus}（全{len(dialogues)}対話中 {len(targets)}対話を処理）")

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        for d in targets:
            n = build_inputs(tokenizer, instructions, d["utterances"])["input_ids"].shape[1]
            print(f"dialogue {d['dialogue_id']}: {len(d['utterances'])}発話, 入力 {n} tokens")
        print(json.dumps(targets[0]["utterances"][:3], ensure_ascii=False, indent=2))
        return

    tokenizer, model = load_model(args.model)
    output_path = Path(args.output).expanduser()

    # 保存形式は elyza_label.py と同じ [dialogue_id, [ラベル配列], dialogue_id, ...]
    all_labels = []
    for d in targets:
        print(f"dialogue {d['dialogue_id']} ({len(d['utterances'])}発話)")
        labels = label_dialogue(
            d["utterances"], tokenizer, model, instructions, args.chunk_size, args.max_retries
        )
        all_labels.append(d["dialogue_id"])
        all_labels.append(labels)
        save_list(all_labels, output_path)
        print(f"  {len(labels)}発話ラベル処理完了")
    print(f"保存完了: {output_path}")


if __name__ == "__main__":
    main()
