from openai import OpenAI
from dotenv import load_dotenv
import os
import time


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

start_time = time.perf_counter()
response = client.responses.create(
    model = "gpt-5.4-mini",
    input="野球について50字程度で解説して"
)
end_time = time.perf_counter()

total_time = end_time - start_time
print(response.output_text)
print(f"実行時間: {total_time:.5f}")