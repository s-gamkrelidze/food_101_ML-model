from src.models_architecture import CustomCNN, EfficientNetClassifier, MobileNetClassifier
from typing import Literal
from torchmetrics import Accuracy, F1Score
from torch import nn
import time
from functools import wraps
import torch.optim as optim


from src.models_architecture import MobileNetClassifier, EfficientNetClassifier, CustomCNN



def load_model_for(model_name: str):
    model_classes = {
        'mobilenet': MobileNetClassifier,
        'efficientnet': EfficientNetClassifier,
        'custom': CustomCNN
    }
    return model_classes[model_name]



def metrics(metric_name: Literal['acc', 'f1_score'], num_class:int):
    metric_class = {
        'acc' : Accuracy(task='multiclass', num_classes=num_class),
        'f1_score' : F1Score(task='multiclass', num_classes=num_class, average='macro',)
    }

    return metric_class[metric_name]


def loss_function(loss_name: Literal['cross_entropy_normal'], loss_label_smoothing):
    loss_class={
        'cross_entropy_normal' :  nn.CrossEntropyLoss(label_smoothing=loss_label_smoothing)
    }

    return loss_class[loss_name]




def optim_function(optim_name: Literal['Adam', 'Adam_w'], 
                   optim_learning_rate,
                   model_param):
   
   
    optim_class={
        'Adam' :  optim.Adam(lr=optim_learning_rate, params=model_param,),
        'Adam_w' : optim.AdamW(lr=optim_learning_rate, params=model_param)
    
    }

    return optim_class[optim_name]






def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Execute the actual function
        result = func(*args, **kwargs)
        
        end_time = time.time()
        print(f"[{func.__name__}] executed in {end_time - start_time:.4f} seconds")
        return result
        
    return wrapper
