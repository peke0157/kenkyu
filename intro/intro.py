from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model = "gpt-5-mini",
    instructions=(
        "あなたはやさしい日本語の先生です。"
        "難しい言葉を避け、1文を短くしてください。"
    ),
    input="人工衛星は地球のまわりを回りながら、地上の様子を観測します"
)
print(response.output_text)