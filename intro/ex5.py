from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
prompt_path = Path("prompts/prompt.txt")
instructions = prompt_path.read_text(encoding="utf-8")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5-mini",
    instructions=instructions,
    input="佐々木選手の進路選択とソフトバンクホークスの入団交渉について、Yahoo!ニュースのコメント欄で話題になっています。コメントでは、MLBドラフトでの指名順位が高くない場合はNPB（特にホークス）入団も現実的な選択肢になるのではないかという意見や、ホークスがポスティング移籍を認めていない方針が入団の障壁になるとの指摘が見られます。また、スタンフォード大学での学業継続や、MLB挑戦を優先するべきか、日本で実績を積んでからメジャーを目指すべきかという進路の悩みに共感する声もありました。一方で、ホークス側がポスティング容認など条件を緩和すれば入団の可能性が高まるのではないかという意見や、球団のドラフト戦略やスカウト体制の見直しを求める声も寄せられています。",
)
print(response.output_text)
