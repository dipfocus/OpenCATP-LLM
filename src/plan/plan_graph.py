import weakref
from bisect import bisect_left
from typing import Any, Dict, Optional

from src.config import DEFAULT_START_TASK_NAME, GlobalMetricsConfig as mcfg
from src.types import TaskName, ModelName, CostInfo

NodeID = int
EdgeID = int


class PlanGraph:
    """
    Graph 统一维护所有 Node 和 Edge 的强引用，并提供增删节点/边的接口。
    """

    name_to_id: Dict[TaskName, NodeID]
    nodes: Dict[NodeID, 'PlanNode']
    edges: Dict[EdgeID, 'PlanEdge']
    _next_node_id: int
    _next_edge_id: int

    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0

        self.nodes = {}
        self.edges = {}
        self.name_to_id = {}

        # 创建一个默认起始节点
        self.add_node(
            task_name=DEFAULT_START_TASK_NAME,
            is_start_point=True
        )

    @property
    def start_node(self):
        return self.nodes[0]

    def add_node(
            self,
            task_name: TaskName,
            *,
            model_name: Optional[ModelName] = None,
            is_start_point: bool = False,
            is_end_point: bool = True,
            is_done: bool = False,
            value: Any = None
    ) -> 'PlanNode':
        """
        在图中创建一个新的节点并返回。
        """
        if task_name in self.name_to_id:
            raise ValueError(f"Node with task name '{task_name}' already exists.")
        node_id = self._next_node_id
        self._next_node_id += 1

        node = PlanNode(
            node_id=node_id,
            task_name=task_name,
            model_name=model_name,
            is_start_point=is_start_point,
            is_end_point=is_end_point,
            is_done=is_done,
            value=value
        )
        # 由 Graph 来设置 Node 的 graph 弱引用
        node.graph = weakref.ref(self)

        # 在 Graph 中登记该 Node
        self.nodes[node_id] = node
        self.name_to_id[task_name] = node_id

        return node

    def get_or_add_node(self, task_name: TaskName) -> 'PlanNode':
        """
        获取指定名称的 Node（若不存在则创建一个新的 Node）。
        """
        node_id = self.name_to_id.get(task_name)
        if node_id is None:
            return self.add_node(task_name)
        return self.nodes[node_id]

    def add_edge(
            self,
            source: 'PlanNode',
            target: 'PlanNode',
            *,
            task_name: Optional[TaskName] = None,
            model_name: Optional[ModelName] = None
    ) -> 'PlanEdge':
        """
        创建一个新的 Edge（从 source 到 target）并加入当前 Graph。
        如果未显式指定 task_name、model_name，则默认从 target 继承。
        """
        edge_id = self._next_edge_id
        self._next_edge_id += 1

        if task_name is None:
            task_name = target.task_name
        if model_name is None:
            model_name = target.model_name

        edge = PlanEdge(
            edge_id=edge_id,
            source=source,
            target=target
        )
        # Edge 内部持有对 Graph 的弱引用
        edge.graph = weakref.ref(self)

        # 双向登记
        self.edges[edge_id] = edge
        source.out_edges[edge_id] = weakref.ref(edge)
        target.in_edges[edge_id] = weakref.ref(edge)

        return edge

    def remove_node(self, node_id: NodeID) -> None:
        """
        从 Graph 中删除指定节点及其所有相关的边。
        """
        node = self.nodes.pop(node_id, None)
        if (not node) or (node.node_id == 0):
            return

        # 同步清理 name_to_id 映射
        if self.name_to_id.get(node.task_name) == node_id:
            self.name_to_id.pop(node.task_name, None)

        # 先把所有与该节点关联的边ID收集起来，避免边遍历/删除冲突
        related_edge_ids = set(node.in_edges.keys()) | set(node.out_edges.keys())

        for e_id in related_edge_ids:
            self.remove_edge(e_id)

    def remove_edge(self, edge_id: EdgeID) -> None:
        """
        从 Graph 中删除指定边，并同步清理涉及到的 source / target 节点对该边的引用。
        """
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return

        source_node = edge.source()
        target_node = edge.target()

        # 从 source_node.out_edges 清理
        if source_node is not None:
            source_node.out_edges.pop(edge_id, None)

        # 从 target_node.in_edges 清理
        if target_node is not None:
            target_node.in_edges.pop(edge_id, None)

    # def __repr__(self) -> str:
    #     """
    #     返回 Graph 的字符串信息：节点ID列表、边ID列表、以及 task_name -> node_id 的映射。
    #     """
    #     cls_name = self.__class__.__name__
    #     return (
    #         f"{cls_name}("
    #         f"nodes={list(self.nodes.keys())}, "
    #         f"edges={list(self.edges.keys())}, "
    #         f"name_to_id={self.name_to_id}"
    #         f")"
    #     )


class PlanNode:
    """
    Node 中保存：自身ID、出/入边(弱引用)、所属 Graph(弱引用)等信息。
    """
    node_id: NodeID
    task_name: TaskName
    model_name: Optional[ModelName]
    is_start_point: bool
    is_end_point: bool
    is_done: bool
    value: Any
    costs: Optional[CostInfo]
    price: Optional[float]
    critical_exec_time: Optional[float]

    graph: Optional[weakref.ReferenceType['PlanGraph']]
    in_edges: Dict[EdgeID, weakref.ReferenceType['PlanEdge']]
    out_edges: Dict[EdgeID, weakref.ReferenceType['PlanEdge']]

    def __init__(
            self,
            node_id: NodeID,
            task_name: TaskName,
            model_name: Optional[ModelName] = None,
            is_start_point: bool = False,
            is_end_point: bool = True,
            is_done: bool = False,
            value: Any = None
    ) -> None:
        self.node_id = node_id
        self.task_name = task_name
        self.model_name = model_name
        self.is_start_point = is_start_point
        self.is_end_point = is_end_point
        self.is_done = is_done
        self.value = value
        self.costs = None
        self.price = None
        self.critical_exec_time = None

        self.graph = None
        self.in_edges = {}
        self.out_edges = {}

    def get_value(self) -> Any:
        """
        获取节点的值。
        """
        return self.value

    def set_value(self, value: Any) -> None:
        """
        设置节点的值。
        """
        self.value = value
        self.is_done = True

    def calculate_price_and_save(self) -> float:
        """
        The tool execution price is calculated by:
        Price = exec_time x (cpu_long_term_mem x cpu_long_term_mem_pricing
                        + cpu_short_term_mem x cpu_short_term_mem_pricing)
                        + (gpu_long_term_mem x gpu_long_term_mem_pricing
                        + gpu_short_term_mem x gpu_short_term_mem_pricing)
                  + price_per_request
        """
        if self.costs is None:
            raise RuntimeError("Costs information is not set for this node.")

        short_term_cpu_price = self.costs.short_term_cpu_memory * mcfg.cpu_short_memory_pricing_per_mb
        short_term_gpu_price = self.costs.short_term_gpu_memory * mcfg.cpu_short_memory_pricing_per_mb
        long_term_cpu_memory = mcfg.tools_gpu_long_term_mem[self.task_name]
        long_term_gpu_memory = mcfg.tools_cpu_long_term_mem[self.task_name]

        long_term_cpu_memory_tiers = sorted(mcfg.cpu_long_memory_pricing.keys())
        long_term_gpu_memory_tiers = sorted(mcfg.gpu_long_memory_pricing.keys())
        long_term_cpu_price_unit = mcfg.cpu_long_memory_pricing[
            bisect_left(long_term_cpu_memory_tiers, long_term_cpu_memory)
        ]
        long_term_gpu_price_unit = mcfg.gpu_long_memory_pricing[
            bisect_left(long_term_gpu_memory_tiers, long_term_gpu_memory)
        ]

        price = self.costs.exec_time * (
                long_term_cpu_memory * long_term_cpu_price_unit
                + self.costs.short_term_cpu_memory * short_term_cpu_price
                + long_term_gpu_memory * long_term_gpu_price_unit
                + self.costs.short_term_gpu_memory * short_term_gpu_price
        ) + mcfg.price_per_request

        self.price = price
        return price

        # def __repr__(self) -> str:
        #     return f"Node(id={self.node_id}, task_name={self.task_name}, model={self.model_name})"


class PlanEdge:
    """
    Edge 中保存：自身ID、源节点/目标节点(弱引用)、所属 Graph(弱引用)等信息。
    """
    edge_id: EdgeID

    graph: Optional[weakref.ReferenceType['PlanGraph']]
    source: weakref.ReferenceType['PlanNode']
    target: weakref.ReferenceType['PlanNode']

    def __init__(
            self,
            edge_id: EdgeID,
            source: 'PlanNode',
            target: 'PlanNode',
    ) -> None:
        self.edge_id = edge_id

        self.graph = None
        self.source = weakref.ref(source)
        self.target = weakref.ref(target)

    # def __repr__(self) -> str:
    #     s_node = self.source()
    #     t_node = self.target()
    #     return (
    #         f"Edge(id={self.edge_id}, "
    #         f"source={s_node.node_id if s_node else None}, "
    #         f"target={t_node.node_id if t_node else None}, "
    #         f"task={self.task_name}, "
    #         f"model={self.model_name})"
    #     )
