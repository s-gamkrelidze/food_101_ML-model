from torch import nn
from torchvision.models import (
    mobilenet_v3_large, 
    MobileNet_V3_Large_Weights,
    efficientnet_v2_s, 
    EfficientNet_V2_S_Weights
) 
from abc import ABC, abstractmethod
from typing import cast



class BaseImageClassifier(nn.Module, ABC):
    def __init__(self, num_classes, dropout: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = self.model_structure()

    @abstractmethod
    def model_structure(self) -> nn.Module:
        pass

    def forward(self, x):
        return self.model(x)


class MobileNetClassifier(BaseImageClassifier):
    def __init__(self, num_classes, pretrained=True, freeze=True, dropout=0.2):
        self.pretrained = pretrained
        self.freeze = freeze
        super().__init__(num_classes=num_classes, dropout=dropout)

    def model_structure(self)-> nn.Module:
        weights = MobileNet_V3_Large_Weights.DEFAULT if self.pretrained else None
        model = mobilenet_v3_large(weights=weights)

        if self.freeze:
            for param in model.features.parameters():
                param.requires_grad = False

        last_linear = cast(nn.Linear, model.classifier[0])
        in_features = last_linear.in_features

        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes)
        )
        return model


class EfficientNetClassifier(BaseImageClassifier):
    def __init__(self, num_classes, pretrained=True, freeze=True, dropout=0.2):
        self.pretrained = pretrained
        self.freeze = freeze
        super().__init__(num_classes=num_classes, dropout=dropout)

    def model_structure(self) -> nn.Module:
        weights = EfficientNet_V2_S_Weights.DEFAULT if self.pretrained else None
        model = efficientnet_v2_s(weights=weights)

        if self.freeze:
            for param in model.features.parameters():
                param.requires_grad = False

        last_linear = cast(nn.Linear, model.classifier[-1])
        in_features = last_linear.in_features

        model.classifier = nn.Sequential(
        nn.Dropout(self.dropout),
        nn.Linear(in_features, self.num_classes)
            )
        return model


    

class CustomCNN(BaseImageClassifier):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__(num_classes=num_classes, dropout=dropout)
    
    def model_structure(self) -> nn.Module:
        features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.num_classes)
        )

        return nn.Sequential(features, classifier)