from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
prompt_path = Path("prompts/keicyo.txt")
instructions = prompt_path.read_text(encoding="utf-8")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_label_data(file_path, file_path1):
    bert_path = "./my_custom_bert"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
        
    
    
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
        load_label_data(label_path, label_path1)
        
        response = client.responses.create(
            model="gpt-5-mini", instructions=instructions, input=user_input
        )
        print(f"GPT:  {response.output_text}")
        conversation_history.append(
            {"role": "assistant", "content": response.output_text}
        )


if __name__ == "__main__":
    chatbot()
    
