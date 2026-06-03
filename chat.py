from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
prompt_path = Path("prompts/chatprompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chatbot():
    print("今の気分はどうですか？？")
    while True:

        conversation_history = []
        user_input = input("User: ")
        # 会話履歴をリストに保存
        if user_input.lower() == "exit":
            print("今日もお疲れ様でした.")
            break
        conversation_history.append({"role": "user", "content": user_input})

        chat_completion = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
        )
        response = chat_completion.choices[0].message.content
        print(f"GPT:  {response}")
        conversation_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chatbot()
