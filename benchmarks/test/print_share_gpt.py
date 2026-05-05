import json
import pandas as pd
from pathlib import Path

DATASETS_PATH = Path(__file__).parent.parent / "datasets"

data_path = DATASETS_PATH / "share_gpt.json"
sgpt_json = pd.read_json(data_path)
print(sgpt_json.head())

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Number of data: {len(data)}")
for i in range(len(data)):
    if len(data[i]['conversations']) % 2 != 0:
        print("find")
    # print(f"id: {data[i]['id']}, length of conversation: {len(data[i]['conversations'])}")

avg_length = 0
for i in range(len(data)):
    avg_length += len(data[i]['conversations'])
avg_length /= len(data)
print(f"average lenght of conversations: {avg_length}")

print("-" * 50)

idx = 0
for i in range(0, len(data[idx]['conversations']), 2):
    print(data[idx]['conversations'][i + 0]['from'], ": ", data[idx]['conversations'][i + 0]['value'])
    print(data[idx]['conversations'][i + 1]['from'], ": ", data[idx]['conversations'][i + 1]['value'])



