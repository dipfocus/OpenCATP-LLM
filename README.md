# OpenCATP-LLM

OpenCATP代码库

## Get Start

1. 安装python，推荐使用版本>=3.12

2. 执行命令：

   ```bash
   pip install -r requirements.txt
   ```

   Note: requirements中标注dev dependencies仅用于开发，如无需debug可不安装

3. 首次运行时，会基于动态导入下载tool hf models至config中标注的hf_cache路径。对于使用github model的任务，需按照对应子目录中的readme文件（可参照运行时错误信息）下载model weight并放入特定文件夹下。

4. 请手动配置config或将数据集放在./dataset路径下。数据集可从[此链接](https://drive.google.com/file/d/1wmxcp-rdxdPgS2UxTXvnQsFgOsrEfmVI/view?usp=drive_link)获取

5. 一个普通测试用例参考test.py

## Source Code

主要代码统一放置在src目录中: 

```
./src
├── config.py
├── data_loader.py
├── __init__.py
├── metrics
│   ├── evaluator.py
│   ├── __init__.py
│   └── runtime_cost.py
├── plan
│   ├── __init__.py
│   ├── plan_graph.py
│   └── plan.py
├── tools
│   ├── github_models
│   ├── grouped_tools.py
│   ├── __init__.py
│   ├── tool_manager.py
│   └── tool.py
├── types.py
└── utils.py
```

关键的问题：

- 代码里含todo的部分
- invalid plan的metric
- 复杂non_seq的结果收集问题。



关键代码部分举例（大部分是ai，**架构为了debug有些延迟循环引用来不及改了，无法进行文档生成**）

## 1. `src/plan/plan.py`

### 主要类

#### `Plan`

- **用途**
   封装了一个执行计划（PlanGraph）及所需的所有工具（Tool），并管理工具的生命周期。提供执行整个计划图、计算价格和执行时间，以及收集最终结果的功能。
- **主要属性**
  - `graph`: `PlanGraph` 对象，表示整个执行流程的有向图。
  - `tools`: 字典，键为 `(TaskName, ModelName)` 元组，值为对应的 `Tool` 实例。
  - `is_done`: 布尔值，表示该计划是否已执行完毕。
  - `price`: `float`，执行所有节点后累计的价格。
  - `exec_time`: `float`，表示执行整条关键路径消耗的时间。
- **主要方法**
  1. `__init__(plan_info: Any = None)`
     - **功能**
        初始化 Plan 对象。如果提供了 `plan_info`，则构建对应的 `PlanGraph`。
     - 参数
       - `plan_info`: 描述任务和依赖关系的结构化信息（类型不固定）。
  2. `create_graph_from_plan_info(plan_info: Any) -> None`
     - **功能**
        从给定的 `plan_info` 数据中解析并创建 `PlanGraph` 的节点与边。
     - 参数
       - `plan_info`: 用于描述任务及其依赖结构的对象。
  3. `prepare_tools() -> None`
     - **功能**
        预加载并分配执行计划所需的所有工具（`Tool`）。如果工具默认是 CPU 模式，则尝试切换到可用的 GPU 上。
  4. `clean_tools() -> None`
     - **功能**
        释放或卸载已用工具，将其移动回 CPU 并清理缓存。
  5. `_execute_on_graph(input_data: Any, cost_aware: bool) -> None`
     - **功能**
        按拓扑顺序或依赖顺序执行图中每个节点的操作，将结果存放到对应节点中。
     - 参数
       - `input_data: Any`: 初始输入数据，用于起始节点。
       - `cost_aware: bool`: 是否进行资源及成本统计。
  6. `collect_results() -> Dict[TaskName, Any]`
     - **功能**
        收集计划中所有终端节点（end-point）的输出，并整合到一个字典里返回。
  7. `calculate_price_and_save() -> float`
     - **功能**
        根据各节点的成本信息和预定义的价格配置，计算出整条执行计划的累计价格，并存入 `self.price`。
  8. `calculate_exec_time_and_save() -> float`
     - **功能**
        通过类似 BFS 的方式计算计划的关键路径总执行时间，并存入 `self.exec_time`。
  9. `execute(input_data: Any, cost_aware: bool = True) -> Any`
     - **功能**
        综合调用一系列方法（`prepare_tools()`、`_execute_on_graph()`、`collect_results()`、`calculate_exec_time_and_save()`、`calculate_price_and_save()` 等），完成计划的整体执行流程并返回结果。
     - 参数
       - `input_data: Any`: 计划执行的起始输入数据。
       - `cost_aware: bool`: 是否收集和计算执行成本。
  10. `cleanup(clean_tools: bool = True) -> None`
      - **功能**
         清理 Plan 对象，释放资源，包括图、工具、价格和执行时间等信息的重置。
      - 参数
        - `clean_tools`: 是否在清理前调用 `clean_tools()`。

------

## 2. `src/plan_graph.py`

### 主要类

#### `PlanGraph`

- **用途**
   维护一个有向图结构，包含对节点和边的维护以及添加、移除节点和边的方法。

- **主要属性**

  - `name_to_id`: 将任务名 (`TaskName`) 映射到节点 ID 的字典。
  - `nodes`: 以 `node_id` 为键，对应 `PlanNode` 实例的字典。
  - `edges`: 以 `edge_id` 为键，对应 `PlanEdge` 实例的字典。

- **主要方法**

  1. `__init__()`

     - **功能**
        初始化图结构，并自动创建一个默认的起始节点（`DEFAULT_START_TASK_NAME`）。

  2. `start_node`

      (property)

     - **功能**
        返回默认的起始节点（通常是 `node_id=0`）。

  3. `add_node(...) -> PlanNode`

     - **功能**
        向图中添加一个新的 `PlanNode`，并注册其 task_name 到 node_id 的映射。
     - 主要参数
       - `task_name: TaskName`: 节点对应的任务名
       - `model_name: Optional[ModelName]`
       - `is_start_point: bool`
       - `is_end_point: bool`
       - ...

  4. `get_or_add_node(task_name: TaskName) -> PlanNode`

     - **功能**
        根据任务名获取已有节点，如不存在则创建新的节点。

  5. `add_edge(source: PlanNode, target: PlanNode) -> PlanEdge`

     - **功能**
        在图中创建一条从 `source` 到 `target` 的有向边，并更新双方状态。

  6. `remove_node(node_id: NodeID) -> None`

     - **功能**
        根据 `node_id` 移除对应节点，并同时移除相关的边。

  7. `remove_edge(edge_id: EdgeID) -> None`

     - **功能**
        根据 `edge_id` 移除对应的边，并更新源节点和目标节点的关系。

#### `PlanNode`

- **用途**
   表示 `PlanGraph` 中的节点，包含任务名、模型名及数据、执行信息等。
- **主要属性**
  - `node_id`: 节点 ID。
  - `task_name`: 任务名称。
  - `model_name`: 可选的模型名称。
  - `is_start_point`: 是否是起始节点。
  - `is_end_point`: 是否是终端节点。
  - `value`: 节点的存储值（执行结果）。
  - `costs`: 节点的执行成本信息（参考 `CostInfo`）。
  - `price`: 节点执行所需的价格。
  - `critical_exec_time`: 节点在关键路径上的累计执行时间。
  - `in_edges`, `out_edges`: 分别存储入边和出边的弱引用字典。
- **主要方法**
  1. `__init__(...)`
     - **功能**
        初始化节点时，设置各种属性并创建空的 `in_edges`、`out_edges`。
  2. `get_value() -> Any`
     - **功能**
        获取节点当前存储的结果。
  3. `set_value(value: Any) -> None`
     - **功能**
        向节点写入结果，并标记节点执行完成。
  4. `calculate_price_and_save() -> float`
     - **功能**
        根据节点的成本信息和配置（`Mcfg`）计算执行价格，存入 `self.price` 并返回。

#### `PlanEdge`

- **用途**
   表示图中一条有向边，存储对源节点和目标节点的弱引用，以及对所属图的引用。
- **主要属性**
  - `edge_id`: 边 ID。
  - `source`: 源节点的弱引用。
  - `target`: 目标节点的弱引用。
- **主要方法**
  - `__init__(edge_id: EdgeID, source: PlanNode, target: PlanNode)`
    - **功能**
       初始化时记录边 ID，以及源、目标节点引用。

------

## 3. `src/tool/tool.py`

### 主要类

#### `Tool`

- **用途**
   封装对单个模型或可执行过程的操作，并可选地进行资源使用监控（如 CPU/GPU 占用、执行时间等）。

- **主要属性**

  - `config: ModelConfig`: 模型的配置信息。
  - `model: Optional[torch.nn.Module]`: PyTorch 模型实例。
  - `process: Optional[Callable[..., Any]]`: 用于执行推理或其他自定义操作的可调用对象。
  - `options: Dict[str, Any]`: 存储初始化时传入的额外选项。
  - `_device: str`: 当前模型所在的设备（`cpu` 或类似 `cuda:0`）。

- **主要方法**

  1. `__init__(config, model, process=None, device='cpu', **kwargs)`

     - **功能**
        初始化时记录模型及其配置，并将模型移动到指定设备，设置为 eval 模式。
     - 参数
       - `config: ModelConfig`
       - `model: torch.nn.Module`
       - `process: Optional[Callable[..., Any]]`
       - `device: str`
       - `kwargs: Any` 其他选项。

  2. `device`

      (property) 及对应的 setter

     - **功能**
        获取或设置模型当前所在的计算设备，并在 setter 中同时移动模型到该设备。

  3. `to(device: str) -> None`

     - **功能**
        手动将模型移动到指定设备。

  4. `execute(*args: Any, cost_aware: bool, **kwargs: Any) -> Any`

     - **功能**
        执行 `process` 函数。如果 `cost_aware` 为真，则进行 CPU/GPU 内存和执行时间的监控，并返回结果和监控数据。
     - 参数
       - `cost_aware: bool`: 是否收集执行的资源成本信息。
       - 其他 `*args` 和 `**kwargs` 会传递给 `process`。

------

## 4. `src/tool/tool_manager.py`

### 主要类

#### `ToolManager`

- **用途**
   根据任务类型（`TaskName`）集中管理不同类型的工具（`Tool`），负责加载、列出、获取指定任务和模型的工具实例。
- **主要属性**
  - `tool_cls_groups: Dict[TaskName, Type[GroupedTools]]`: 用于映射任务名称到对应的工具组类。
  - `tool_groups: Dict[TaskName, GroupedTools]`: 运行时存储已实例化的工具组。
- **主要方法**
  1. `__init__()`
     - **功能**
        初始化一个空的 `tool_groups` 字典。
  2. `load_model(task_name: TaskName, model_name: ModelName) -> None`
     - **功能**
        加载指定任务和模型对应的工具组，在内部完成模型的初始化或缓存。
  3. `load_models(task_name: TaskName = 'all_tasks', model_name: ModelName = 'all_models') -> None`
     - **功能**
        根据参数决定一次性加载所有任务/所有模型，或只加载指定任务/模型。
  4. `list_models() -> Dict[TaskName, List[ModelName]]`
     - **功能**
        汇总 `tool_groups` 中所有已加载的模型信息，返回按任务分类的模型名称列表。
  5. `get_model(task_name: TaskName, model_name: ModelName) -> Tool`
     - **功能**
        获取已加载的指定任务、指定模型的 `Tool` 实例。如果 `model_name` 为空，则自动使用默认模型（在 `MODEL_REGISTRY` 中的第一个）。

------

## 5. `src/dataloader.py`

### 主要类

#### `TaskDataset`

- **用途**
   一个 PyTorch 风格的 `Dataset`，用于加载图片和文本任务数据，数据按 `sample_id` 编排。
- **主要属性**
  - `input_data`: 存储所有输入的字典（键为 `sample_id`，值为对应的图像或文本张量/内容）。
  - `output_data`: 存储对应输出的字典。
  - `sample_ids`: 有序的样本 ID 列表，用于索引。
- **主要方法**
  1. `__init__(data_path: str, *, task_id: int)`
     - **功能**
        根据给定的 `data_path` 和 `task_id` 初始化并加载对应任务的数据。
     - 参数
       - `data_path: str`: 数据根目录。
       - `task_id: int`: 任务 ID，用于拼接成具体路径。
  2. `_load_images(image_dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor]]`
     - **功能**
        从指定路径加载图像文件，转换为张量并按 `sample_id` 存储。
  3. `_load_text(file_path: str) -> Dict[SampleID, Dict[str, TextContent]]`
     - **功能**
        从文本文件加载每行内容，并按行号映射到 `sample_id`。
  4. `_load_files(dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor | TextContent]]`
     - **功能**
        识别指定目录下的 `images` 文件夹或 `.txt` 文件，并调用相应方法加载数据，最终按样本 ID 整合。
  5. `_load_data() -> None`
     - **功能**
        根据输入、输出路径分别读取数据，并更新 `input_data` 和 `output_data`。
  6. `__getitem__(index: int) -> Dict[str, int | Dict[str, torch.Tensor | TextContent]]`
     - **功能**
        按索引返回一个样本的完整数据，包括样本 ID、输入和输出。
  7. `__len__() -> int`
     - **功能**
        返回数据集的样本数量。
