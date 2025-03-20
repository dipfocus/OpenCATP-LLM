import gc
from collections import deque
from typing import Any, Dict, Set, Deque, Tuple

import torch

from src.config import TOOL_DEVICE_LIST
from src.tools import tool_manager, Tool
from src.types import TaskName, ModelName, CostInfo
from src.utils import get_available_device
from plan_graph import NodeID, PlanNode, PlanGraph


class Plan:
    graph: PlanGraph
    tools: Dict[Tuple[TaskName, ModelName], Tool]
    is_executed: bool
    price: float
    exec_time: float

    def __init__(self, description: Any) -> None:
        self.graph = PlanGraph()
        self.tools = {}
        self.is_done = False
        self.price = 0.0
        self.exec_time = 0.0
        if description:
            self.create_graph_from_description(description)

    def create_graph_from_description(self, description: Any) -> None:
        """
        Create a plan graph from a description.
        """
        # ['Image Deblurring', ['Input query'], 'Image Denoising', ['Image Deblurring'], 'Colorization',
        #  ['Image Denoising'], 'Image Classification', ['Colorization'], 'Visual Question Anwsering',
        #  [['Input question'], ['Image Deblurring']], 'Machine Translation', ['Visual Question Anwsering']]
        for i in range(0, len(description), 2):
            task_name = description[i]
            dependencies = description[i + 1]
            assert len(dependencies) >= 1, f"Now only one dependency supported in each operation."

            source = self.graph.add_node(task_name)
            target = self.graph.get_or_add_node(dependencies[0])
            self.graph.add_edge(source, target)

    def prepare_tools(self) -> None:
        """
        Prepare tools for the plan graph.
        """
        for node in self.graph.nodes.values():
            task_name = node.task_name
            model_name = node.model_name

            if (task_name, model_name) not in self.tools:
                self.tools[(task_name, model_name)] = tool_manager.get_model(task_name, model_name)
            tool = self.tools[(task_name, model_name)]
            if tool.device == 'cpu':
                device = get_available_device(TOOL_DEVICE_LIST)
                tool.to(device)

    def clean_tools(self) -> None:
        """
        Unload all tools in the plan graph.
        """
        used_devices = set()
        for tool in self.tools.values():
            if tool.device != 'cpu':
                used_devices.add(tool.device)
                tool.to('cpu')

        for device in used_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        gc.collect()

    def _execute_on_graph(self, input_data: Any, *, method='bfs') -> Any:
        # Assuming the plan graph is given in the form of a topologically sorted list of nodes and edges,
        # then we can use BFS to execute the plan.
        if method == 'bfs':
            visited: Set[NodeID] = set()
            queue: Deque[PlanNode] = deque()

            start_node = self.graph.start_node
            start_node.set_value(input_data)
            start_node.costs = CostInfo(
                exec_time=0.0,
                short_term_cpu_memory=0.0,
                short_term_gpu_memory=0.0,
            )
            visited.add(start_node.node_id)
            queue.append(start_node)

            while queue:
                current_node = queue.popleft()
                if current_node != start_node:
                    current_input = {}
                    for _, edge_ref in current_node.in_edges.items():
                        edge = edge_ref()
                        source_node = edge.source()
                        current_input.update(source_node.get_value())

                    tool = self.tools[(current_node.task_name, current_node.model_name)]
                    result, costs = tool.execute(current_input, cost_aware=True)
                    current_node.set_value(result)
                    current_node.costs = costs

                for _, edge_ref in current_node.out_edges.items():
                    edge = edge_ref()
                    target_node = edge.target()
                    if target_node.node_id not in visited:
                        visited.add(target_node.node_id)
                        queue.append(target_node)

    def collect_results(self) -> Dict[TaskName, Any]:
        """
        Collect results from the plan graph.
        """
        results = {}
        for node in self.graph.nodes.values():
            if node.is_end_point:
                results.update(node.get_value())
        return results

    def calculate_price_and_save(self):
        price = 0.0
        for node in self.graph.nodes.values():
            if node.costs is None:
                node.calculate_price_and_save()
            price += node.price

        self.price = price
        return price

    def calculate_exec_time_and_save(self):
        exec_time = 0.0

        visited: Set[NodeID] = set()
        queue: Deque[PlanNode] = deque()

        start_node = self.graph.start_node
        start_node.critical_exec_time = 0.0
        visited.add(start_node.node_id)
        queue.append(start_node)

        while queue:
            current_node = queue.popleft()
            max_exec_time_before = 0.0
            for _, edge_ref in current_node.in_edges.items():
                edge = edge_ref()
                source_node = edge.source()
                max_exec_time_before = max(max_exec_time_before, source_node.critical_exec_time)

            current_node.critical_exec_time = current_node.costs.exec_time + max_exec_time_before

            for _, edge_ref in current_node.out_edges.items():
                edge = edge_ref()
                target_node = edge.target()
                if target_node.node_id not in visited:
                    visited.add(target_node.node_id)
                    queue.append(target_node)

        self.exec_time = exec_time
        return exec_time

    def execute(self, input_data: Any) -> Any:
        self.prepare_tools()
        self._execute_on_graph(input_data, method='bfs')
        self.collect_results()
        self.clean_tools()
