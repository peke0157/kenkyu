import torch
from pathlib import Path
import json 
from llama_cpp import Llama

prompt_path = Path("prompts/keicyo.txt")
instructions = prompt_path.read_text(encoding="utf-8")

llm = Llama.from_pretrained(
    model_path="elyza/Llama-3-ELYZA-JP-8B-GGUF",
    chat_format="Llama-3",
    n_ctx=1024,
)

label_path = Path("outputs/label_output.json")
label1_path = Path("outputs/label_output_101-200.json")

def load_label_data(file_path, file1_path):
    with open(file_path, "r", encoding="utf-8") as f:
        file_data = json.load(f)
    with open(file1_path, "r", encoding="utf-8") as f1:
        file1_data = json.load(f1)
    return file_data, file1_data


def chatbot():
    print("今日はどんな1日でしたか？")
    
    conversation_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "fin":
            print("今日もお疲れ様でした。")
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        load_label_data()
        
        llm.create_chat_completion(
            message = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input}
            ]
            
        )

