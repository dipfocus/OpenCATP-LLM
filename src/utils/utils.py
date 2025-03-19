import re

import torch

from src.config import TOOL_GPU_MEMORY_ALLOC_LIMIT


def get_available_device(device_list):
    """
    返回列表中第一个显存剩余超过 3GB 的 CUDA 设备，
    如果都不满足则返回 "cpu"。

    参数：
        device_list (list of str): 设备列表，例如 ['cuda:0', 'cuda:1']，也可以为空

    返回：
        str: 满足条件的 CUDA 设备，或 "cpu"
    """

    def qualifies(device):
        free_mem, _ = torch.cuda.mem_get_info(torch.device(device))
        return free_mem > TOOL_GPU_MEMORY_ALLOC_LIMIT

    return next((d for d in device_list if qualifies(d)), "cpu")


def normalize_name(source: str):
    source = source.strip()
    source = source.replace(" ", "-").replace('_', "-")
    return re.sub(r'(?<!^)(?=[A-Z])', '-', source).lower()
