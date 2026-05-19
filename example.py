from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model = "gpt-5.5",
    input="「今日は雨です」を、年上の人に向けた日本語で言い換えてください。"
)

print(response.output_text)