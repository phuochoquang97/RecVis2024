from transformers import Dinov2ForImageClassification
from transformers.activations import GELUActivation
import torch.nn as nn


class DinoV2Model:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model = self._init_model()

    def _init_model(self):
        model = Dinov2ForImageClassification.from_pretrained(
            "facebook/dinov2-large-imagenet1k-1-layer"
        )

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1000),
            GELUActivation(),
            nn.Dropout(0.3),
            nn.Linear(1000, self.num_classes),
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

        return model

    def get_model(self):
        return self.model
