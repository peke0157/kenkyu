import json
from pathlib import Path
import os
import ast

label_path = Path("../outputs/ans_label.json")



with label_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
    print(len(data))

    
    
c_list = []
c1_list = []

correct = 0
uncorrect = 0

new_list = []
for dialogue in data:
    if isinstance(dialogue, list):
        new_list.append(dialogue)


for item in new_list:
    data_list = [test["self_disclosure"] for test in item]
    print(data_list)
    c_list.append(data_list)
    for i in range(len(data_list)):
        if data_list[i] == '1':
            correct += 1
        else:
            uncorrect += 1
            
print(f"自己開示有り{correct}")
print(f"自己開示無し{uncorrect}")


count = 0
count1 = 0


for item_11 in c_list:
    if isinstance(item_11, list):
        count += 1
        
    
for item_12 in c1_list:
    if isinstance(item_12, list):
        count1 += 1
    
print(count)
print(count1)


