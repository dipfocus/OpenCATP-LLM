from dataclasses import dataclass
from typing import Tuple, Any, Dict, Literal

ArgsType = (Tuple, Dict[str, Any])
TaskName = Literal[
    'sentiment_analysis',
    'machine_translation',
    'image_classification',
    'object_detection',
    'image_super_resolution',
    'image_colorization',
    'image_denoising',
    'image_deblurring',
    'image_captioning',
    'text_to_image',
    'question_answering',
    'visual_question_answering',
    'text_summarization',
    'text_generation',
    'mask_filling'
]
ModelName = str

@dataclass
class CostInfo:
    """
    CostInfo is a dictionary that contains the cost information of a tool.
    """
    exec_time: float
    short_term_cpu_memory: float
    short_term_gpu_memory: float
