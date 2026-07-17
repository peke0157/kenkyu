import os
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

prompt_path = Path("../prompts/keicyo.txt")
base_prompt = prompt_path.read_text(encoding="utf-8")


bert_path = ".././my_custom_bert"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
bert_model.eval()


def chatbot():
    print("今の気分はどうですか？？")

    conversation_history = []
    while True:
        user_input = input("User: ")
        # 会話履歴をリストに保存
        if user_input.lower() == "exit":
            print("今日もお疲れ様でした.")
            break
        conversation_history.append({"role": "user", "content": user_input})

        # 入力文の自己開示判定
        inputs = bert_tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            print(prediction)
            print(outputs.logits)
            is_bert_judge = prediction == 1

        if is_bert_judge:
            judge_prompt = "自己開示有り"
        else:
            judge_prompt = "自己開示無し"
        system_prompt = f"{base_prompt}\n : {judge_prompt}"
        print(judge_prompt)


if __name__ == "__main__":
    chatbot()
