import json
from pathlib import Path

save_path = Path("../outputs/ans_label.json")

with open("../outputs/label_output.json", "r", encoding="utf-8") as f:
    file_data = json.load(f)

with open("../outputs/label_output_101-200.json", "r", encoding="utf-8") as f1:
    file_data1 = json.load(f1)

new_list = file_data + file_data1

with save_path.open("w", encoding="utf-8") as f:
    json.dump(new_list, f, ensure_ascii=False, indent=4)
    
print("保存完了")