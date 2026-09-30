"""傾聴チャットの修正版。既存の chat_keicyo.py は変更しない。"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import APIError, OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "keicyo_binary.md"
BERT_PATH = BASE_DIR / "my_custom_bert"
THRESHOLD = 0.7
MAX_HISTORY_TURNS = 10
OPENAI_MODEL = "gpt-5.6-luna"

# test/bertcheck.py と同じ対応を採用。学習時のラベル定義が分かれば照合する。
NO_DISCLOSURE_CLASS = 0
DISCLOSURE_CLASS = 1


def classify_disclosure(text, tokenizer, model):
    max_length = model.config.max_position_embeddings
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]

    if probabilities[DISCLOSURE_CLASS].item() >= THRESHOLD:
        return "自己開示有り"
    if probabilities[NO_DISCLOSURE_CLASS].item() >= THRESHOLD:
        return "自己開示無し"
    return "自己開示不明"


def chatbot():
    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    base_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    if model.config.num_labels != 2:
        raise ValueError("自己開示の分類モデルは2クラスである必要があります。")
    model.eval()
    client = OpenAI()

    history = []
    print("今の気分はどうですか？？")
    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\n今日もお疲れ様でした。")
            break

        if user_input.strip().lower() == "exit":
            print("今日もお疲れ様でした。")
            break
        if not user_input.strip():
            continue

        judge_prompt = classify_disclosure(user_input, tokenizer, model)
        messages = history + [{"role": "user", "content": user_input}]
        instructions = f"{base_prompt}\n\n今回のユーザー発話の自己開示判定: {judge_prompt}"

        try:
            response = client.responses.create(
                model=OPENAI_MODEL,
                instructions=instructions,
                input=messages,
            )
        except APIError as exc:
            print(f"応答を取得できませんでした: {exc}")
            continue

        answer = response.output_text.strip()
        if not answer:
            print("応答が空だったため、もう一度入力してください。")
            continue

        print(f"GPT:  {answer}")
        history = (
            messages + [{"role": "assistant", "content": answer}]
        )[-2 * MAX_HISTORY_TURNS :]


if __name__ == "__main__":
    chatbot()
