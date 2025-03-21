import gc
from collections import deque
from typing import Any, Dict, Set, Deque, Tuple

import torch

from src.config import TOOL_DEVICE_LIST
from src.tools import tool_manager, Tool
from src.types import TaskName, ModelName, CostInfo
from src.utils import get_available_device
from .plan_graph import NodeID, PlanNode, PlanGraph


class Plan:
    """
    A Plan object that encapsulates a PlanGraph along with the tools needed to execute it.
    It manages the lifecycle of the tools (prepare & clean) and provides methods to execute
    the plan graph (e.g., BFS) and collect final results.
    """

    graph: PlanGraph
    tools: Dict[Tuple[TaskName, ModelName], Tool]
    is_done: bool
    price: float
    exec_time: float

    def __init__(self, description: Any = None) -> None:
        """
        Initialize the Plan with a PlanGraph and optional description to build the graph structure.

        Args:
            description: A structure describing how to build the plan graph.
        """
        self.graph = PlanGraph()
        self.tools = {}
        self.is_done = False
        self.price = 0.0
        self.exec_time = 0.0

        # If description is provided, build the graph from it.
        if description:
            self.create_graph_from_description(description)

    def create_graph_from_description(self, description: Any) -> None:
        """
        Create a plan graph from a description object (e.g., a list describing tasks and dependencies).

        The format might be something like:
            [task_name, [dependency_task_name], task_name, [dependency_task_name], ...]
        Example:
            ['Image Deblurring', ['Input query'], 'Image Denoising', ['Image Deblurring'], ...]

        Currently, an assert enforces that each task has at least one dependency.
        """
        for i in range(0, len(description), 2):
            task_name = description[i]
            dependencies = description[i + 1]

            assert len(dependencies) >= 1, (
                "At least one dependency required per operation."
            )

            source_node = self.graph.add_node(task_name)

            for dependency in dependencies:
                target_node = self.graph.get_or_add_node(dependency)
                self.graph.add_edge(source_node, target_node)

    def prepare_tools(self) -> None:
        """
        Prepare (load and allocate) all tools required by the plan.
        Moves tools to an available device if they are CPU-based by default.
        """
        for node in self.graph.nodes.values():
            task_name = node.task_name
            model_name = node.model_name

            tool_key = (task_name, model_name)
            if tool_key not in self.tools:
                self.tools[tool_key] = tool_manager.get_model(task_name, model_name)

            tool = self.tools[tool_key]
            # If the tool is CPU-based, we attempt to move it to an available device (GPU) if possible
            if tool.device == 'cpu':
                device = get_available_device(TOOL_DEVICE_LIST)
                tool.to(device)

    def clean_tools(self) -> None:
        """
        Release or unload all tools used in the plan.
        Moves them back to 'cpu', clears CUDA caches, and forces garbage collection.
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

    def _execute_on_graph(self, input_data: Any, method: str = 'bfs') -> None:
        """
        Note: Assuming the plan graph is given in the form of a topologically sorted list of nodes and edges,
        then we can use BFS to execute the plan.

        Execute the plan graph with the given input data.
        By default, a BFS approach is used to traverse and compute node outputs.

        Args:
            input_data: The initial input to be stored in the start node.
            method: The traversal method, default is 'bfs'.
        """
        if method == 'bfs':
            visited: Set[NodeID] = set()
            queue: Deque[PlanNode] = deque()

            # Set the input data in the start node
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
                # Skip the start_node processing since it already holds the initial input
                if current_node != start_node:
                    current_input = {}

                    for _, edge_ref in current_node.in_edges.items():
                        edge = edge_ref()
                        source_node = edge.source()
                        current_input.update(source_node.get_value())

                    tool_key = (current_node.task_name, current_node.model_name)
                    tool = self.tools[tool_key]

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
        Collect results from all end-point nodes in the plan graph.
        Returns a dictionary combining the results of all end-point nodes.
        """
        results = {}
        for node in self.graph.nodes.values():
            if node.is_end_point:
                node_value = node.get_value()
                if isinstance(node_value, dict):
                    results.update(node_value)
                else:
                    results[node.task_name] = node_value

        return results

    def calculate_price_and_save(self) -> float:
        """
        Calculate the total price for executing all nodes in the plan,
        based on each node's cost metrics and pricing configuration in PlanNode.
        """
        price_sum = 0.0
        for node in self.graph.nodes.values():
            if node.costs is None:
                # If no cost info, attempt to calculate price anyway (may raise error internally)
                node.calculate_price_and_save()
            price_sum += node.price or 0.0

        self.price = price_sum
        return price_sum

    def calculate_exec_time_and_save(self) -> float:
        """
        Calculate the total (critical path) execution time of the plan using BFS-like traversal.
        This sets each node's 'critical_exec_time' to the sum of its own exec_time and
        the maximum exec_time among its predecessors.
        """
        exec_time_total = 0.0

        visited: Set[NodeID] = set()
        queue: Deque[PlanNode] = deque()

        start_node = self.graph.start_node
        start_node.critical_exec_time = 0.0
        visited.add(start_node.node_id)
        queue.append(start_node)

        while queue:
            current_node = queue.popleft()

            # The maximum of all parent nodes' critical_exec_time
            max_parent_time = 0.0
            for _, edge_ref in current_node.in_edges.items():
                edge = edge_ref()
                source_node = edge.source()
                if source_node and source_node.critical_exec_time is not None:
                    max_parent_time = max(max_parent_time, source_node.critical_exec_time)

            # Add this node's own time
            node_exec_time = current_node.costs.exec_time if current_node.costs else 0.0
            current_node.critical_exec_time = node_exec_time + max_parent_time

            exec_time_total = max(exec_time_total, current_node.critical_exec_time)

            for _, edge_ref in current_node.out_edges.items():
                edge = edge_ref()
                target_node = edge.target()
                if target_node.node_id not in visited:
                    visited.add(target_node.node_id)
                    queue.append(target_node)

        self.exec_time = exec_time_total
        return exec_time_total

    def execute(self, input_data: Any) -> Any:
        """
        Prepare tools, execute the plan (currently with BFS), collect results, then clean up.
        Returns the collected results from all end-point nodes.
        """
        self.prepare_tools()
        self._execute_on_graph(input_data, method='bfs')
        results = self.collect_results()
        self.clean_tools()
        return results
