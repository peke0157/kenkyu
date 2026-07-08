from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()
prompt_path = Path("prompts/kyokan.txt")
base_prompt = prompt_path.read_text(encoding="utf-8")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


bert_path = "./my_custom_bert"
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
        inputs = bert_tokenizer(user_input, return_tensors = "pt")
        with torch.no_grad():
            outputs = bert_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            is_bert_judge = (prediction == 1)
            
            
        if is_bert_judge:
            judge_prompt = "自己開示有り"
        else:
            judge_prompt = "自己開示無し"
        system_prompt = f"{base_prompt}\n : {judge_prompt}" 
        response = client.responses.create(
            model="gpt-5-mini", instructions=system_prompt, input=conversation_history
        )
        print(f"GPT:  {response.output_text}")
        conversation_history.append(
            {"role": "assistant", "content": response.output_text}
        )


if __name__ == "__main__":
    chatbot()
    
