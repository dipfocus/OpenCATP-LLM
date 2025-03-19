from typing import Any, Dict, List, Optional, Callable
import gc
import time

import psutil
import torch

from src.config import log, cfg
from src.model_registry import ModelConfig, MODEL_REGISTRY
from src.types import TaskName, ModelName, CostInfo
from src.utils.cost import CPUMemoryMonitor


class Tool:
    config: ModelConfig
    model: Optional[Any]
    process: Optional[Callable]
    options: Dict[str, Any]
    device: Optional[str]

    def __init__(self, config, model, *, process=None, device='cpu', **kwargs):
        self.config = config
        self.model = model
        self.process = process
        self.device = device
        self.options = kwargs

        if self.device != 'cpu':
            self.model.to(self.device)
        self.model.eval()

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    def execute(self, *args, cost_aware: bool, **kwargs) -> Any:
        if not cost_aware:
            return self.process(*args, **kwargs, device=self.device)

        # Cleanup for cost measurement
        if self.device != 'cpu':
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()

        # use time.perf_counter(), CPUMemoryMonitor(), torch.cuda.max_memory_allocated() respectively
        cpu_mem_monitor = CPUMemoryMonitor(interval=0.1)
        time_before = time.perf_counter() * 1000
        cpu_mem_before = psutil.Process().memory_info().rss / (1024 ** 2)
        gpu_mem_before = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.device != 'cpu' else 0

        cpu_mem_monitor.start()
        result = self.process(*args, **kwargs, device=self.device)
        if self.device != 'cpu':
            torch.cuda.synchronize()
        cpu_mem_monitor.stop()

        time_after = time.perf_counter() * 1000
        cpu_mem_after = cpu_mem_monitor.get_max_cpu_memory_allocated(size='MB')
        gpu_mem_after = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.device != 'cpu' else 0

        costs = {
            'exec_time': time_after - time_before,
            'cpu_short_term_mem': cpu_mem_after - cpu_mem_before,
            'gpu_short_term_mem': gpu_mem_after - gpu_mem_before
        }

        return result, costs

    def __repr__(self):
        return repr(self.config)
