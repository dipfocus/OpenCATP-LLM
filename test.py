import json

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np

from src.config import DATA_PATH
from src.plan import Plan
from src.utils import print_graph
from src.data_loader import TaskDataset
from src.tools import tool_manager
from src.metrics import get_vit_score, get_bert_score, calculate_qop

with open("./train_data_plan_map.json", "r") as f:
    data = json.load(f)

for task_info in data.values():
    if task_info["task_id"] != 201:
        continue
    task_id = task_info["task_id"]
    description = task_info["task_desc_raw"]
    plans = task_info["plans"]
    break

# tool_manager.load_models()

plan = Plan(plans[0])
graph = plan.graph
data_set = TaskDataset(DATA_PATH, task_id=task_id)
data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
for batch in data_loader:
    print_graph(graph, save_path='./graph.png')
    sample_id = batch["sample_id"]
    input_data = batch["input"]
    output_data = batch["output"]
    result = plan.execute(input_data)
    if 'image' in result:
        vit_score = get_vit_score(result['image'], output_data['image'])
    if 'text' in result:
        bert_score = get_bert_score(result['text'], output_data['text'])
    print(vit_score, bert_score)

print("done")