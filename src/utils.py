from typing import cast

import torch

from src.config import TOOL_GPU_MEMORY_ALLOC_LIMIT, DEFAULT_START_TASK_NAME
from src.types import TaskName


def get_available_device(device_list):
    """
    Return the first device in the list that has enough free memory.
    If none, return "cpu".

    Args:
        device_list: List of CUDA devices to choose from.

    Returns:
        str: The device name that has enough free memory or "cpu".
    """

    def qualifies(device):
        free_mem, _ = torch.cuda.mem_get_info(torch.device(device))
        return free_mem > TOOL_GPU_MEMORY_ALLOC_LIMIT

    return next((d for d in device_list if qualifies(d)), "cpu")


def normalize_task_name(source: str) -> TaskName:
    source = source.strip()
    source = source.replace(" ", "_").lower()

    # Normalize special cases
    # fixme: name alias for backward compatibility
    match source:
        case "input_of_query":
            source = DEFAULT_START_TASK_NAME
        case 'input_query':
            source = DEFAULT_START_TASK_NAME
        case 'colorization':
            source = 'image_colorization'
        case _:
            pass
    target = cast(TaskName, source)
    return target
