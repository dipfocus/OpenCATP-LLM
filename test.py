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
from src.metrics.evaluator import get_image_similarity, get_bert_score, calculate_qop

with open("./train_data_plan_map.json", "r") as f:
    data = json.load(f)

for task_info in data.values():
    if task_info["task_id"] != 201:
        continue
    task_id = task_info["task_id"]
    description = task_info["task_desc_raw"]
    plans = task_info["plans"]
    break

plan = Plan(plans[0])
graph = plan.graph
data_set = TaskDataset(DATA_PATH, task_id=task_id)
data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
for batch in data_loader:
    print_graph(graph, save_path='./graph.png')
    sample_id = batch["sample_id"]
    input_data = batch["input"]
    output_data = batch["output"]
    plan.execute(input_data)
    break
