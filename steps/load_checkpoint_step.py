from pathlib import Path

import torch
from torch import nn
from zenml import step

from src.params import load_model_for


@step(enable_cache=False)
def load_checkpoint_model(checkpoint_path: Path, model_name: str) -> nn.Module:
    """Load a saved checkpoint into the matching PyTorch architecture."""
    model_class = load_model_for(model_name=model_name)

    if model_name == "custom":
        model = model_class(num_classes=101, dropout=0.2)
    else:
        model = model_class(
            num_classes=101,
            pretrained=False,
            freeze=True,
            dropout=0.2,
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.eval()
