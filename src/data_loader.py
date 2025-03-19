import os
from collections import defaultdict
from typing import Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore

SampleID = int
TextContent = str


class TaskDataset(Dataset):
    input_data: Dict[SampleID, Dict[str, torch.Tensor | TextContent]]
    output_data: Dict[SampleID, Dict[str, torch.Tensor | TextContent]]

    def __init__(self, data_path: str, task_id: int):
        self.task_data_path = os.path.join(data_path, str(task_id))
        self._load_data_()

    @staticmethod
    def _load_images(image_dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor]]:
        result = {}
        image_processor = transforms.Compose([
            transforms.PILToTensor(),
        ])
        for image_path in os.listdir(image_dir_path):
            sample_id = int(os.path.splitext(image_path)[0])
            with Image.open(os.path.join(image_dir_path, image_path)) as image:
                result[sample_id] = {'image': image_processor(image.convert("RGB").copy())}
        return result

    @staticmethod
    def _load_text(file_path: str) -> Dict[SampleID, Dict[str, TextContent]]:
        result = {}
        with open(file_path) as f:
            text_type = os.path.splitext(file_path)[0]
            text = f.readlines()
            for sample_id, line in enumerate(text):
                result[sample_id] = {text_type: line.strip()}
        return result

    def _load_files(self, dir_path: str) -> Dict[SampleID, Dict[str, torch.Tensor | TextContent]]:
        result = defaultdict(dict)
        for path in os.listdir(dir_path):
            if path == 'images':
                file_data = self._load_images(os.path.join(dir_path, path))
            elif path.endswith('.txt'):
                file_data = self._load_text(os.path.join(dir_path, path))
            else:
                raise ValueError(f'Unknown file type: {path}')

            for sample_id, data in file_data.items():
                result[sample_id].update(data)

        return dict(result)

    def _load_data_(self) -> None:
        input_path = os.path.join(self.task_data_path, 'inputs')
        output_path = os.path.join(self.task_data_path, 'outputs')
        self.input_data = self._load_files(input_path)
        self.output_data = self._load_files(output_path)

    def __getitem__(self, sample_id: SampleID) -> Dict[str, SampleID | Dict[str, torch.Tensor | TextContent]]:
        data_item = {
            'sample_id': sample_id,
            'input': self.input_data[sample_id],
            'output': self.output_data[sample_id]
        }
        return data_item

    def __len__(self) -> int:
        return len(self.input_data)
