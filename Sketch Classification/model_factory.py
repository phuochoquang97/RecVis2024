"""Python file to instantiate the model and the transform that goes with it."""

from data import data_transforms
from model import Net
from dinov2_model import DinoV2Model  # Import the new DinoV2 model wrapper
import torch.nn as nn


class ModelFactory:
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "dinov2":
            return DinoV2Model(self.num_classes).get_model()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name in ["basic_cnn", "dinov2"]:
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
