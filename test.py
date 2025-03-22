import json
from src.plan import Plan
from src.utils import print_graph
from src.data_loader import TaskDataset

with open('./train_data_plan_map.json', 'r') as f:
    data = json.load(f)

for task_info in data.values():
    task_id = task_info['task_id']
    description = task_info['task_desc_raw']
    plans = task_info['plans']
    break

plan = Plan(plans[0])
graph = plan.graph
data = TaskDataset('./dataset', task_id=task_id)
