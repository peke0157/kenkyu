"""既存の自己開示プロンプトを使い、Ollamaで対話にラベルを付ける。"""

import argparse
import json
import logging
import os
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "../prompts/self-disclosure_level.md"
SAVE_PATH = BASE_DIR / "../outputs/llama_label_output.json"
CORPUS_PATH = BASE_DIR / "../corpus/japanese-daily-dialogue/data/topic1.json"
MODEL = "llama3.2:3b"
LOGGER = logging.getLogger(__name__)


def make_response_schema(utterance_count):
    return {
        "type": "array",
        "minItems": utterance_count,
        "maxItems": utterance_count,
        "items": {
            "type": "object",
            "properties": {
                "turn_num": {"type": "integer", "minimum": 1},
                "speaker": {"type": "string"},
                "utterance": {"type": "string"},
                "self_disclosure": {"type": "string", "enum": list("01234")},
            },
            "required": ["turn_num", "speaker", "utterance", "self_disclosure"],
            "additionalProperties": False,
        },
    }


def validate_utterances(utterances):
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("utterancesには1件以上の発話配列が必要です。")
    seen = set()
    for item in utterances:
        if not isinstance(item, dict):
            raise ValueError("各発話はJSONオブジェクトにしてください。")
        turn = item.get("turn_num")
        if type(turn) is not int or turn < 1 or turn in seen:
            raise ValueError("turn_numは重複のない1以上の整数にしてください。")
        seen.add(turn)
        if not all(isinstance(item.get(key), str) for key in ("speaker", "utterance")):
            raise ValueError("speakerとutteranceには文字列が必要です。")


def validate_result(result, utterances):
    if not isinstance(result, list) or len(result) != len(utterances):
        raise ValueError("出力の発話数が入力と一致しません。")
    expected_keys = {"turn_num", "speaker", "utterance", "self_disclosure"}
    for original, labeled in zip(utterances, result):
        if not isinstance(labeled, dict) or set(labeled) != expected_keys:
            raise ValueError("出力のキーが指定形式と一致しません。")
        if type(labeled["turn_num"]) is not int:
            raise ValueError("出力のturn_numが整数ではありません。")
        if any(
            labeled[key] != original[key]
            for key in ("turn_num", "speaker", "utterance")
        ):
            raise ValueError("出力の発話内容・順序が入力と一致しません。")
        if not isinstance(labeled["self_disclosure"], str) or labeled[
            "self_disclosure"
        ] not in list("01234"):
            raise ValueError("self_disclosureは文字列の0～4にしてください。")


def judge_self_disclosure(
    text, client, instructions, model, n_ctx, n_batch, max_tokens
):
    validate_utterances(text)
    input_data = [
        {key: item[key] for key in ("turn_num", "speaker", "utterance")}
        for item in text
    ]
    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
        ],
        format=make_response_schema(len(text)),
        options={
            "temperature": 0,
            "num_ctx": n_ctx,
            "num_batch": n_batch,
            "num_predict": max_tokens,
        },
        stream=False,
    )
    if response.get("done_reason") == "length":
        raise RuntimeError(
            "出力が途中で終了しました。LLAMA_N_TOKENSとLLAMA_N_CTXを調整してください。"
        )
    content = response["message"]["content"]
    LOGGER.debug("モデルの生出力: %r", content)
    result = json.loads(content)
    validate_result(result, text)
    LOGGER.info(
        "%.2fs (入力: %s tokens, 出力: %s tokens)",
        time.perf_counter() - start,
        response.get("prompt_eval_count", "?"),
        response.get("eval_count", "?"),
    )
    return result


def label_dataset(utterances, **kwargs):
    result = judge_self_disclosure(utterances, **kwargs)
    LOGGER.info("%d発話ラベル処理完了", len(utterances))
    # ELYZA版と同じ、対話内の発話配列を包む形式。
    return [result]


def save_list(label_list, save_path=SAVE_PATH):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = save_path.with_name(save_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(label_list, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    temporary_path.replace(save_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=SAVE_PATH)
    parser.add_argument("--model", default=os.getenv("LLAMA_MODEL", MODEL))
    parser.add_argument(
        "--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        n_ctx = int(os.getenv("LLAMA_N_CTX", "4096"))
        n_batch = int(os.getenv("LLAMA_N_BATCH", "512"))
        max_tokens = int(os.getenv("LLAMA_N_TOKENS", "2048"))
        if n_ctx < 2048 or n_batch < 1 or not 0 < max_tokens < n_ctx:
            raise ValueError(
                "LLAMA_N_CTXは2048以上、LLAMA_N_BATCHは1以上、LLAMA_N_TOKENSは1以上かつコンテキスト長未満にしてください。"
            )
        if args.limit < 1 or not args.model.strip():
            raise ValueError("--limitは1以上、--modelは空でない名前にしてください。")
        instructions = args.prompt.read_text(encoding="utf-8")
        if not instructions.strip():
            raise ValueError("プロンプトが空です。")
        data = json.loads(args.corpus.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("コーパスは1件以上の対話を含むJSON配列にしてください。")
        selected = data[: args.limit]
        for dialogue in selected:
            if not isinstance(dialogue, dict) or "dialogue_id" not in dialogue:
                raise ValueError("各対話にはdialogue_idが必要です。")
            validate_utterances(dialogue.get("utterances"))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        from ollama import Client
    except ImportError:
        parser.error(
            "ollamaが必要です。python -m pip install -r ../requirements.txt を実行してください。"
        )

    client = Client(host=args.host, timeout=600)
    all_labels = []
    for dialogue in selected:
        dialogue_id = dialogue["dialogue_id"]
        LOGGER.info("対話ID: %s", dialogue_id)
        try:
            label_list = label_dataset(
                dialogue["utterances"],
                client=client,
                instructions=instructions,
                model=args.model.strip(),
                n_ctx=n_ctx,
                n_batch=n_batch,
                max_tokens=max_tokens,
            )
            all_labels.append(dialogue_id)
            all_labels.extend(label_list)
            save_list(all_labels, args.output)
        except Exception as exc:
            LOGGER.error(
                "対話ID %sで停止: %s（それまでの保存済み結果は保持されます）",
                dialogue_id,
                exc,
            )
            return 1
    LOGGER.info("保存完了: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
