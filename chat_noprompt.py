from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

        response = client.responses.create(model="gpt-5-mini", input=user_input)
        print(f"GPT:  {response.output_text}")
        conversation_history.append(
            {"role": "assistant", "content": response.output_text}
        )


if __name__ == "__main__":
    chatbot()
