"""ELYZA-JP-8B（llama.cpp）によるベースライン対話、自己開示判定モデル無し"""

import os
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from llama_cpp import Llama
import json

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR.parent / "prompts" / "chatprompt.md"
SAVE_OUTPUT_PATH = BASE_DIR.parent / "outputs" / "chat_output.json"
MAX_HISTORY_TURNS = 10

MODEL_REPO = "elyza/Llama-3-ELYZA-JP-8B-GGUF"
MODEL_FILENAME = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"

LLAMA_OPTIONS = {
    "chat_format": "llama-3",
    "n_ctx": 4096,
    "n_threads": 6,
    "n_gpu_layers": -1,  # CPU版では無視される。GPUサーバーで実験するとき用
    "seed": 42,
    "use_mmap": True,
    "verbose": False,
}
GENERATION_OPTIONS = {
    "max_tokens": 256,
    "temperature": 0.6,
    "top_p": 0.9,
}


def load_llm():
    model_path = os.getenv("LLAMA_MODEL_PATH")
    if model_path is None:
        cached_path = try_to_load_from_cache(MODEL_REPO, MODEL_FILENAME)
        model_path = cached_path if isinstance(cached_path, str) else None
    if model_path is not None:
        return Llama(model_path=model_path, **LLAMA_OPTIONS)
    return Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        **LLAMA_OPTIONS,
    )


def generate_reply(llm, system_prompt, messages):
    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_prompt}, *messages],
        **GENERATION_OPTIONS,
    )
    return response["choices"][0]["message"]["content"].strip()


def make_save_schema(utterance):
    """会話履歴を保存する"""
    return {
        "type": "array",
        "minItems": utterance,
        "maxItems": utterance,
        "items": {
            "type": "object",
            "properties": {
                "dialogues_id": {
                    "type": "integer",
                    "minimum": 1,
                },
                "turn_num": {
                    "type": "integer",
                    "minimum": 1,
                },
            },
        },
    }


def chatbot():
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    llm = load_llm()

    history = []
    print("今の気分はどうですか？？")
    while True:
        try:
            user_input = input("Seeker: ")
        except (EOFError, KeyboardInterrupt):
            print("\n今日もお疲れ様でした。")
            break

        if user_input.strip().lower() == "exit":
            print("今日もお疲れ様でした。")
            break
        if not user_input.strip():
            continue

        messages = history + [{"role": "user", "content": user_input}]
        try:
            answer = generate_reply(llm, system_prompt, messages)
        except ValueError as exc:  # コンテキスト長超過など
            print(f"応答を生成できませんでした: {exc}")
            continue

        if not answer:
            print("応答が空だったため、もう一度入力してください。")
            continue

        print(f"Supporter:  {answer}")
        history = (messages + [{"role": "assistant", "content": answer}])[
            -2 * MAX_HISTORY_TURNS :
        ]


if __name__ == "__main__":
    chatbot()
