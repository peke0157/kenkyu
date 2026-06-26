import pandas as pd

df = pd.read_json("../outputs/label_output.json", encoding="utf-8")
df_1 = pd.read_json("../outputs/label_output_101-200.json", encoding="utf-8")

df.to_csv("../outputs/label_output.csv", encoding="utf-8")
df_1.to_csv("../outputs/label_output_101-200.csv", encoding="utf-8")