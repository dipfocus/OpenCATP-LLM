from typing import Dict, List, Type

from src.config import MODEL_REGISTRY, log
from src.types import TaskName, ModelName
from grouped_tools import (
    Tool,
    GroupedTools,
    SentimentAnalysisTools,
    MachineTranslationTools,
    ImageClassificationTools,
    ObjectDetectionTools,
    ImageSuperResolutionTools,
    ImageColorizationTools,
    ImageDenoisingTools,
    ImageDeblurringTools,
    ImageCaptioningTools,
    TextToImageTools,
    QuestionAnsweringTools,
    VisualQuestionAnsweringTools,
    TextSummarizationTools,
    TextGenerationTools,
    MaskFillingTools
)


class ToolManager:
    model_cls_groups: Dict[TaskName, Type[GroupedTools]] = {
        'sentiment_analysis': SentimentAnalysisTools,
        'machine_translation': MachineTranslationTools,
        'image_classification': ImageClassificationTools,
        'object_detection': ObjectDetectionTools,
        'image_super_resolution': ImageSuperResolutionTools,
        'image_colorization': ImageColorizationTools,
        'image_denoising': ImageDenoisingTools,
        'image_deblurring': ImageDeblurringTools,
        'image_captioning': ImageCaptioningTools,
        'text_to_image': TextToImageTools,
        'question_answering': QuestionAnsweringTools,
        'visual_question_answering': VisualQuestionAnsweringTools,
        'text_summarization': TextSummarizationTools,
        'text_generation': TextGenerationTools,
        'mask_filling': MaskFillingTools,
    }
    model_groups: Dict[TaskName, GroupedTools]

    def __init__(self):
        self.model_groups = {}

    def load_model_(self, task_name: TaskName, model_name: ModelName) -> None:
        if task_name not in self.model_groups:
            model_cls = self.model_cls_groups[task_name]
            self.model_groups[task_name] = model_cls()
        self.model_groups[task_name].load_model_(model_name)

    def load_models_(self, task_name: TaskName = 'all_tasks', model_name: ModelName = 'all_models') -> None:
        if task_name == 'all_tasks':
            if model_name == 'all_models':
                for task_name_item in MODEL_REGISTRY:
                    log.info('Initializing all models for task: {}'.format(task_name_item))
                    for model_name_item in MODEL_REGISTRY[task_name_item]:
                        self.load_model_(task_name_item, model_name_item)
            else:
                raise ValueError('Task_name is "all_tasks" but model_name is specified.')
        else:
            if model_name == 'all_models':
                for model_name_item in MODEL_REGISTRY[task_name]:
                    self.load_model_(task_name, model_name_item)
            else:
                self.load_model_(task_name, model_name)

    def list_models(self) -> Dict[TaskName, List[ModelName]]:
        result: Dict[TaskName, List[ModelName]] = {}
        for task_name in self.model_groups:
            result.update(self.model_groups[task_name].list_models())
        return result

    def get_model(self, task_name: TaskName, model_name: ModelName) -> Tool:
        if model_name is None:
            # If model_name is None, return the default model.
            available_models = list(MODEL_REGISTRY[task_name].keys())
            if len(available_models) == 0:
                raise ValueError(f"No available models for task {task_name}.")

            log.info('No model_name specified in {}, using the default model: {}'.format(
                task_name, available_models[0]
            ))
            model_name = available_models[0]

        if task_name not in self.model_groups:
            model_cls = self.model_cls_groups[task_name]
            self.model_groups[task_name] = model_cls()
        return self.model_groups[task_name].get_model(model_name)
