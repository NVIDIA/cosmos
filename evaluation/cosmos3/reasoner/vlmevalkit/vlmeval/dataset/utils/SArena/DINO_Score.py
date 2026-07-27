import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .base_metric import BaseMetric


class DINOScoreCalculator(BaseMetric):
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self.get_DINOv2_model("base")
        self.model = self.model.to(self.device)
        self.metric = self.calculate_DINOv2_similarity_score

    def get_DINOv2_model(self, model_size):
        model_map = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
        }
        name = model_map.get(model_size)
        if not name:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")

        model = AutoModel.from_pretrained(name)
        processor = AutoImageProcessor.from_pretrained(name)

        return model, processor

    def process_input(self, image, processor):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(image, torch.Tensor):
            features = image.unsqueeze(0) if image.dim() == 1 else image
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")
        return features

    def calculate_DINOv2_similarity_score(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('pred_im')
        features1 = self.process_input(image1, self.processor)
        features2 = self.process_input(image2, self.processor)

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(features1, features2).item()
        sim = (sim + 1) / 2

        return sim
