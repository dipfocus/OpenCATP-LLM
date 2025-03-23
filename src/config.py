import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from loguru import logger as log

from .types import TaskName, ModelName

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
home_dir = os.path.dirname(src_dir)

# Constants that should typically be reset based on the device environment
DATA_PATH = "/home/data/OpenCATP_LLM/dataset"
PRETRAINED_LLM_DIR = "/home/data/pretrained_llms/"

TOOL_DEVICE_LIST = ["cuda:0", "cuda:1"]
EVALUATOR_DEVICE_LIST = ["cuda:2", "cuda:3"]
# the minimum GPU memory allocation limit, default is 3GB
TOOL_GPU_MEMORY_ALLOC_LIMIT = 3 * 1024 ** 3

# log config when using loguru
LOG_FORMAT_CONSOLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    " | <level>{level:<7}</level>"
    " | <cyan>{name}</cyan>:<cyan>{line}</cyan>"
    " - <level>{message}</level>"
)

# todo: remove this before release
API_BASE = "https://api.chatanywhere.tech/v1"
API_KEY = "sk-CelumY6pSozc9ZVHQSbjdVNk10LxRONiIBo5JxgRUxHShq5z"

DEFAULT_START_TASK_NAME = "input"

OLD_KEYS = [
    "Image Classification",
    "Colorization",
    "Object Detection",
    "Image Deblurring",
    "Image Denoising",
    "Image Super Resolution",
    "Image Captioning",
    "Text to Image Generation",
    "Visual Question Answering",
    "Sentiment Analysis",
    "Question Answering",
    "Text Summarization",
    "Text Generation",
    "Machine Translation",
    "Fill Mask",
]

@dataclass
class GlobalPathConfig:
    hf_cache = os.path.join(home_dir, "hf_cache/")
    data_path = os.path.join(home_dir, "dataset/")
    result_path = os.path.join(home_dir, "results/")


@dataclass
class GlobalTaskConfig:
    default_train_tasks = [
        1, 4, 5, 6, 7, 9, 12, 13, 14, 19, 20, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43, 44, 46,
        47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 63, 64, 65, 67, 68, 71, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86,
        87, 88, 90, 91, 92, 94, 95, 97, 98, 99, 101, 103, 114, 112, 107,
    ]
    default_eval_tasks = [
        2, 3, 11, 15, 17, 21, 22, 28, 42, 58, 61, 66, 69, 70, 78, 100, 102, 104, 109,
    ]
    default_test_tasks = [
        0, 8, 10, 16, 18, 27, 29, 39, 45, 49, 55, 62, 75, 76, 84, 89, 93, 96, 108, 110, 111,
    ]


@dataclass
class GlobalMetricsConfig:
    MIN_SCORE = 0
    MAX_SCORE = 1
    MIN_COST = 0
    MAX_COST = 0.5669374317944147
    ALPHA = 0.5

    score_penalty = -2  # penalty assigned to the scores of invalid plans
    cost_penalty = 2  # penalty assigned to the costs of invalid plans

    tools_cpu_long_term_mem = {
        "image_classification": 1788.50390625,
        "image_colorization": 1626.96875,
        "object_detection": 1684.6875,
        "image_deblurring": 1693.83984375,
        "image_denoising": 1690.15625,
        "image_super_resolution": 1540.8515625,
        "image_captioning": 2449.8359375,
        "text_to_image": 6746.109375,
        "visual_question_answering": 1953.0234375,
        "sentiment_analysis": 1719.6875,
        "question_answering": 1696.2734375,
        "text_summarization": 3321.2578125,
        "text_generation": 1937.03125,
        "machine_translation": 2388.72265625,
        "mask_filling": 1712.41796875,
    }
    tools_gpu_long_term_mem = {
        "image_classification": 330.2294921875,
        "image_colorization": 131.31689453125,
        "object_detection": 234.7177734375,
        "image_deblurring": 99.74462890625,
        "image_denoising": 99.67431640625,
        "image_super_resolution": 47.52490234375,
        "image_captioning": 937.234375,
        "text_to_image": 5252.0234375,
        "visual_question_answering": 449.14599609375,
        "sentiment_analysis": 256.49755859375,
        "question_answering": 249.19384765625,
        "text_summarization": 1550.06689453125,
        "text_generation": 487.46875,
        "machine_translation": 850.3095703125,
        "mask_filling": 256.61376953125,
    }

    # Long-term cpu memory pricing tiers. Data format: {memory_size(MB): price(USD)} per ms
    cpu_long_memory_pricing = {
        128: 0.0000000021,
        512: 0.0000000083,
        1024: 0.0000000167,
        1536: 0.0000000250,
        2048: 0.0000000333,
        3072: 0.0000000500,
        4096: 0.0000000667,
        5120: 0.0000000833,
        6144: 0.0000001000,
        7168: 0.0000001167,
        8192: 0.0000001333,
        9216: 0.0000001500,
        10240: 0.0000001667,
    }
    # Short-term cpu memory pricing per MB.
    cpu_short_memory_pricing_per_mb = 0.0000000000000302
    # Note that the AWS Lambda does not provide pricing strategy for GPU resources.
    # We use set the GPU prices as three times of CPU prices, according to the following article:
    # https://news.rice.edu/news/2021/rice-intel-optimize-ai-training-commodity-hardware
    gpu_long_memory_pricing = {k: v * 3 for k, v in cpu_long_memory_pricing.items()}
    gpu_short_memory_pricing_per_mb = cpu_short_memory_pricing_per_mb * 3
    price_per_request = 0.0000002

    # tool_prices = pickle.load(open(result_path + 'all_tools_prices.pkl', 'rb'))


@dataclass
class GlobalToolConfig:
    tool_token_start = 80000
    sop_token = tool_token_start
    eop_token = sop_token + 1

    dependency_token_start = 90000
    sod_token = dependency_token_start
    eod_token = sod_token + 1

    max_num_tokens = 50
    max_num_generated_tokens = 40
    max_ep_len = 100  # max length of episode (= max number of tokens to be generated)

    # tool-token mapping
    tool_token_vocabulary = {
        "image_classification": sop_token + 2,
        "image_colorization": sop_token + 3,
        "object_detection": sop_token + 4,
        "image_deblurring": sop_token + 5,
        "image_denoising": sop_token + 6,
        "image_super_resolution": sop_token + 7,
        "image_captioning": sop_token + 8,
        "text_to_image": sop_token + 9,
        "visual_question_answering": sop_token + 10,
        "sentiment_analysis": sop_token + 11,
        "question_answering": sop_token + 12,
        "text_summarization": sop_token + 13,
        "text_generation": sop_token + 14,
        "machine_translation": sop_token + 15,
        "mask_filling": sop_token + 16,
    }
    tool_token_vocabulary_reverse = {v: k for k, v in tool_token_vocabulary.items()}

    # dependency-token mapping
    dependency_token_vocabulary = {
        "image_classification": sod_token + 2,
        "image_colorization": sod_token + 3,
        "object_detection": sod_token + 4,
        "image_deblurring": sod_token + 5,
        "image_denoising": sod_token + 6,
        "image_super_resolution": sod_token + 7,
        "image_captioning": sod_token + 8,
        "text_to_image": sod_token + 9,
        "visual_question_answering": sod_token + 10,
        "sentiment_analysis": sod_token + 11,
        "question_answering": sod_token + 12,
        "text_summarization": sod_token + 13,
        "text_generation": sod_token + 14,
        "machine_translation": sod_token + 15,
        "mask_filling": sod_token + 16,
        "Input of Query": sod_token + 17,
    }
    dependency_token_vocabulary_reverse = {
        v: k for k, v in dependency_token_vocabulary.items()
    }

    tool_io_dict = {
        "image_colorization": ["image", "image"],
        "image_denoising": ["image", "image"],
        "image_deblurring": ["image", "image"],
        "image_super_resolution": ["image", "image"],
        "image_classification": ["image", "text"],
        "image_captioning": ["image", "text"],
        "object_detection": ["image", "text"],
        "text_summarization": ["text", "text"],
        "text_generation": ["text", "text"],
        "machine_translation": ["text", "text"],
        "mask_filling": ["text", "text"],
        "sentiment_analysis": ["text", "text"],
        "text_to_image": ["text", "image"],
        "question_answering": ["text-text", "text"],
        "visual_question_answering": ["image-text", "text"],
    }

    tool_token_io_dict = {
        tool_token_vocabulary["image_colorization"]: ["image", "image"],
        tool_token_vocabulary["image_denoising"]: ["image", "image"],
        tool_token_vocabulary["image_deblurring"]: ["image", "image"],
        tool_token_vocabulary["image_super_resolution"]: ["image", "image"],
        tool_token_vocabulary["image_classification"]: ["image", "text"],
        tool_token_vocabulary["image_captioning"]: ["image", "text"],
        tool_token_vocabulary["object_detection"]: ["image", "text"],
        tool_token_vocabulary["text_summarization"]: ["text", "text"],
        tool_token_vocabulary["text_generation"]: ["text", "text"],
        tool_token_vocabulary["machine_translation"]: ["text", "text"],
        tool_token_vocabulary["mask_filling"]: ["text", "text"],
        tool_token_vocabulary["sentiment_analysis"]: ["text", "text"],
        tool_token_vocabulary["text_to_image"]: ["text", "image"],
        tool_token_vocabulary["question_answering"]: ["text-text", "text"],
        tool_token_vocabulary["visual_question_answering"]: ["image-text", "text"],
    }

    tool_io_dict_collection = {
        "in:image-out:image": [
            "image_colorization",
            "image_denoising",
            "image_deblurring",
            "image_super_resolution",
        ],
        "in:image-out:text": [
            "image_classification",
            "image_captioning",
            "object_detection",
        ],
        "in:text-out:text": [
            "text_summarization",
            "text_generation",
            "machine_translation",
            "mask_filling",
            "sentiment_analysis",
        ],
        "in:text-out:image": ["text_to_image"],
        "in:image,text-out:text": ["visual_question_answering"],
        "in:text,text-out:text": ["question_answering"],
    }
    # same as above, but using tokens
    tool_token_io_dict_collection = {
        "in:image-out:image": [
            tool_token_vocabulary["image_colorization"],
            tool_token_vocabulary["image_denoising"],
            tool_token_vocabulary["image_deblurring"],
            tool_token_vocabulary["image_super_resolution"],
        ],
        "in:image-out:text": [
            tool_token_vocabulary["image_classification"],
            tool_token_vocabulary["image_captioning"],
            tool_token_vocabulary["object_detection"],
        ],
        "in:text-out:text": [
            tool_token_vocabulary["text_summarization"],
            tool_token_vocabulary["text_generation"],
            tool_token_vocabulary["machine_translation"],
            tool_token_vocabulary["mask_filling"],
            tool_token_vocabulary["sentiment_analysis"],
        ],
        "in:text-out:image": [tool_token_vocabulary["text_to_image"]],
        "in:image,text-out:text": [tool_token_vocabulary["visual_question_answering"]],
        "in:text,text-out:text": [tool_token_vocabulary["question_answering"]],
    }

    task_io_dict = {
        "in:image-out:image": set(range(0, 15)),
        "in:image-out:text": set(range(15, 105)),
        "in:text-out:image": set(range(105, 108)),
        "in:text-out:text": set(range(108, 126)),
        "in:image,text-out:text": set(range(126, 171)),
        "in:text,text-out:text": set(range(171, 188)),
    }

    tool_dependencies = {
        # e.g. 'image_colorization': ['image_super_resolution', ...] means that image_colorization depends on image_super_resolution
        "image_colorization": [
            "image_super_resolution",
            "image_deblurring",
            "image_denoising",
        ],
        "image_super_resolution": [
            "image_colorization",
            "image_deblurring",
            "image_denoising",
        ],
        "image_deblurring": [
            "image_colorization",
            "image_super_resolution",
            "image_denoising",
        ],
        "image_denoising": [
            "image_colorization",
            "image_super_resolution",
            "image_deblurring",
        ],
        "image_captioning": [],
        "image_classification": [],
        "object_detection": [],
        "machine_translation": [],
        "sentiment_analysis": [],
        "text_summarization": [],
        "mask_filling": [],
        "text_generation": [],
        "text_to_image": [],
    }
    tool_dependencies_reverse = {
        "image_colorization": [
            "image_super_resolution",
            "image_deblurring",
            "image_denoising",
        ],
        "image_super_resolution": [
            "image_colorization",
            "image_deblurring",
            "image_denoising",
        ],
        "image_deblurring": [
            "image_colorization",
            "image_super_resolution",
            "image_denoising",
        ],
        "image_denoising": [
            "image_colorization",
            "image_super_resolution",
            "image_deblurring",
        ],
        "image_captioning": [],
        "image_classification": [],
        "object_detection": [],
        "machine_translation": [],
        "sentiment_analysis": [],
        "text_summarization": [],
        "mask_filling": [],
        "text_generation": [],
        "text_to_image": [],
    }


@dataclass
class GlobalDataConfig:
    image_sizes = [490 * 402, 582 * 578, 954 * 806, 1921 * 2624]
    text_lengths = [149, 2009, 4464, 7003]


@dataclass
class ModelConfig:
    task_name: Literal[
        "sentiment_analysis",
        "image_classification",
        "image_colorization",
        "object_detection",
        "image_super_resolution",
        "image_captioning",
        "text_to_image",
        "question_answering",
        "text_summarization",
        "text_generation",
        "visual_question_answering",
        "machine_translation",
        "mask_filling",
        "image_deblurring",
        "image_denoising",
    ]
    model_name: str
    source: Literal["huggingface", "github"]
    hf_url: Optional[str]


MODEL_REGISTRY: Dict[TaskName, Dict[ModelName, ModelConfig]] = {
    "sentiment_analysis": {
        "distilbert-sst2": ModelConfig(
            task_name="sentiment_analysis",
            model_name="distilbert-sst2",
            source="huggingface",
            hf_url="distilbert-base-uncased-finetuned-sst-2-english",
        )
    },
    "image_classification": {
        "vit-base": ModelConfig(
            task_name="image_classification",
            model_name="vit-base",
            source="huggingface",
            hf_url="google/vit-base-patch16-224",
        )
    },
    "image_colorization": {
        "siggraph17": ModelConfig(
            task_name="image_colorization",
            model_name="siggraph17",
            source="github",
            hf_url=None,
        )
    },
    "object_detection": {
        "detr-resnet-101": ModelConfig(
            task_name="object_detection",
            model_name="detr-resnet-101",
            source="huggingface",
            hf_url="facebook/detr-resnet-101",
        )
    },
    "image_super_resolution": {
        "swin2sr": ModelConfig(
            task_name="image_super_resolution",
            model_name="swin2sr",
            source="huggingface",
            hf_url="caidas/swin2SR-classical-sr-x2-64",
        )
    },
    "image_captioning": {
        "vit-gpt2": ModelConfig(
            task_name="image_captioning",
            model_name="vit-gpt2",
            source="huggingface",
            hf_url="nlpconnect/vit-gpt2-image-captioning",
        )
    },
    # "text_to_image": {
    #     "stable-diffusion-v1-4": ModelConfig(
    #         task_name="text_to_image",
    #         model_name="stable-diffusion-v1-4",
    #         source="huggingface",
    #         hf_url="CompVis/stable-diffusion-v1-4",
    #     )
    # },
    # "question_answering": {
    #     "distilbert-squad": ModelConfig(
    #         task_name="question_answering",
    #         model_name="distilbert-squad",
    #         source="huggingface",
    #         hf_url="distilbert-base-cased-distilled-squad",
    #     )
    # },
    "text_summarization": {
        "bart-cnn": ModelConfig(
            task_name="text_summarization",
            model_name="bart-cnn",
            source="huggingface",
            hf_url="facebook/bart-large-cnn",
        )
    },
    "text_generation": {
        "gpt2-base": ModelConfig(
            task_name="text_generation",
            model_name="gpt2-base",
            source="huggingface",
            hf_url="gpt2",
        )
    },
    "visual_question_answering": {
        "vilt-vqa": ModelConfig(
            task_name="visual_question_answering",
            model_name="vilt-vqa",
            source="huggingface",
            hf_url="dandelin/vilt-b32-finetuned-vqa",
        )
    },
    "machine_translation": {
        "t5-base": ModelConfig(
            task_name="machine_translation",
            model_name="t5-base",
            source="huggingface",
            hf_url="t5-base",
        )
    },
    "mask_filling": {
        "distilbert-mlm": ModelConfig(
            task_name="mask_filling",
            model_name="distilbert-mlm",
            source="huggingface",
            hf_url="distilbert-base-uncased",
        )
    },
    "image_deblurring": {
        "restormer-deblur": ModelConfig(
            task_name="image_deblurring",
            model_name="restormer-deblur",
            source="github",
            hf_url=None,
        )
    },
    "image_denoising": {
        "restormer-denoise": ModelConfig(
            task_name="image_denoising",
            model_name="restormer-denoise",
            source="github",
            hf_url=None,
        )
    },
}

log.remove()
log.add(sys.stdout, level="INFO", colorize=True, format=LOG_FORMAT_CONSOLE)
log.add(
    f'{current_file_path}/../../logs/{datetime.now().strftime("%Y-%m-%d")}.log',
    level="DEBUG",
)
