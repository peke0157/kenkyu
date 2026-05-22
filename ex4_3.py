from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5-mini",
    instructions="1文だけで要約してください",
    input="西武・渡部聖弥外野手（23）が22日からのオリックス3連戦（ベルーナ）中に三塁守備に復帰する。4月17日の日本ハム戦で右手親指に死球を受け、以降はDHで出場を続けていたが回復。練習では送球も問題なく、渡部は「自分の中ではもう大丈夫です」と話し、3連戦中に相手投手との兼ね合いなどもみて起用される。19日に復帰3戦目となる3軍巨人戦（Gタウン）でフル出場した桑原将志外野手（32）は、26日からの交流戦（ヤクルト戦）で満を持して復帰する見込みとなっている。",
)
print(response.output_text)
