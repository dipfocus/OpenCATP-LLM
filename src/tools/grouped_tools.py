from abc import abstractmethod
from typing import Any, Dict, List
from importlib import import_module

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DetrImageProcessor,
    DetrForObjectDetection,
    ViTFeatureExtractor,
    ViTForImageClassification,
    AutoImageProcessor,
    Swin2SRForImageSuperResolution,
    ViltProcessor,
    ViltForQuestionAnswering,
    VisionEncoderDecoderModel
)

from tool import Tool
from src.config import log, GlobalPathConfig, ModelConfig, MODEL_REGISTRY
from src.types import TaskName, ModelName


class GroupedTools:
    models: Dict[ModelName, Tool]
    task_name: TaskName

    def __init__(self):
        self.models = {}

    def get_model(self, model_name: ModelName) -> Tool:
        if model_name not in self.models:
            self.load_model_(model_name)
        return self.models[model_name]

    def list_models(self) -> Dict[TaskName, List[ModelName]]:
        result = {self.task_name: list(self.models.keys())}
        return result

    def _get_model_config(self, model_name: ModelName) -> ModelConfig:
        if not self.task_name or self.task_name not in MODEL_REGISTRY:
            raise ValueError(f"Models of Task {self.task_name} is not implemented.")
        elif model_name not in MODEL_REGISTRY[self.task_name]:
            raise ValueError(f"Model {model_name} is not implemented for {self.task_name}.")
        return MODEL_REGISTRY[self.task_name][model_name]

    @abstractmethod
    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        ...


class SentimentAnalysisTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'sentiment_analysis'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "distilbert-sst2":
                tokenizer = AutoTokenizer.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = AutoModelForSequenceClassification.from_pretrained(model_config.hf_url,
                                                                           cache_dir=GlobalPathConfig.hf_cache)

                def process(input_data: Any, device) -> Any:
                    inputs = tokenizer(input_data.text, return_tensors="pt", padding=True).to(device)
                    model_output = model(**inputs)
                    pred_ids = torch.argmax(model_output.logits, dim=1)
                    pred_labels = [model.config.id2label[pred_id] for pred_id in pred_ids]
                    return {
                        'pred_labels': pred_labels
                    }

                self.models[model_name] = Tool(model_config, model, tokenizer=tokenizer, process=process)


class MachineTranslationTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'machine_translation'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "t5-base":
                tokenizer = AutoTokenizer.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, tokenizer=tokenizer)


class ImageClassificationTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'image_classification'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "vit-base":
                processor = ViTFeatureExtractor.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = ViTForImageClassification.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, processor=processor)


class ObjectDetectionTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'object_detection'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "detr-resnet-101":
                processor = DetrImageProcessor.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = DetrForObjectDetection.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, processor=processor)


class ImageSuperResolutionTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'image_super_resolution'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "swin2sr":
                processor = AutoImageProcessor.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = Swin2SRForImageSuperResolution.from_pretrained(model_config.hf_url,
                                                                       cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, processor=processor)


class ImageColorizationTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'image_colorization'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "siggraph17":
                colorizers = import_module('github_models.colorization.colorizers')
                model = colorizers.siggraph17()
                self.models[model_name] = Tool(model_config, model)
                # todo: check if model is callable


class ImageDenoisingTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'image_denoising'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "restormer-denoise":
                # params = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                #           'num_refinement_blocks': 4, 'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
                #           'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
                # weights_path = "github_models/Restormer/Denoising/pretrained_models/real_denoising.pth"
                # model = RestormerDenoise(**model_config.hf_init_kwargs)
                # self.models[model_name] = ModelContainer(model_config, model)
                # todo: check if model is callable
                pass


class ImageDeblurringTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'image_deblurring'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "restormer-deblur":
                # params = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                #           'num_refinement_blocks': 4, 'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
                #           'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
                # weights_path = "github_models/Restormer/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth"
                # model = RestormerDeblur(**model_config.hf_init_kwargs)
                # self.models[model_name] = ModelContainer(model_config, model)
                # todo: check if model is callable
                pass


class ImageCaptioningTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'image_captioning'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "vit-gpt2":
                processor = ViTFeatureExtractor.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = VisionEncoderDecoderModel.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, processor=processor)


class TextToImageTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'text_to_image'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "stable-diffusion-v1-4":
                # model = StableDiffusionPipeline.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                # self.models[model_name] = ModelContainer(model_config, model)
                # todo: check if model is callable
                pass


class QuestionAnsweringTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'question_answering'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "distilbert-squad":
                tokenizer = AutoTokenizer.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = AutoModelForQuestionAnswering.from_pretrained(model_config.hf_url,
                                                                      cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, tokenizer=tokenizer)


class VisualQuestionAnsweringTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'visual_question_answering'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "vilt-vqa":
                processor = ViltProcessor.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = ViltForQuestionAnswering.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, processor=processor)


class TextSummarizationTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'text_summarization'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "bart-cnn":
                tokenizer = AutoTokenizer.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, tokenizer=tokenizer)


class TextGenerationTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'text_generation'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "gpt2-base":
                tokenizer = AutoTokenizer.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = AutoModelForCausalLM.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, tokenizer=tokenizer)


class MaskFillingTools(GroupedTools):
    def __init__(self):
        super().__init__()
        self.task_name: TaskName = 'mask_filling'

    def load_model_(self, model_name: ModelName, **kwargs) -> None:
        if model_name in self.models:
            return
        log.info(f"Task: {self.task_name}, loading model: {model_name}")
        model_config = self._get_model_config(model_name)
        match model_name:
            case "distilbert-mlm":
                tokenizer = AutoTokenizer.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                model = AutoModelForMaskedLM.from_pretrained(model_config.hf_url, cache_dir=GlobalPathConfig.hf_cache)
                self.models[model_name] = Tool(model_config, model, tokenizer=tokenizer)
