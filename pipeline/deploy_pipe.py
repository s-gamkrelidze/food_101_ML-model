from pathlib import Path
from typing import Literal

from zenml import pipeline

from src.depl_functions import (
    class_names_step,
    deployment_trigger,
    get_last_model_path,
)
from steps.load_checkpoint_step import load_checkpoint_model
from steps.publish_step import publish_accepted_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_ROOT / "data" / "food-101" / "images"
SAVED_MODELS_PATH = PROJECT_ROOT / "saved_models"


@pipeline(enable_cache=False, name="food101_deployment")
def run_deployment_pipe(
    model_name: str,
    min_accuracy: float = 0.7,
    data_folder_path: str = str(IMAGES_ROOT),
) -> tuple[str, float, float, float]:

    
    """Evaluate the last saved model under <saved_models/model_name> and publish it to <best_models> only if accepted."""

    checkpoint_path = get_last_model_path(
        model_path=str(SAVED_MODELS_PATH / model_name),
    )
   
    approved = deployment_trigger(
        model_name=model_name,
        min_accuracy=min_accuracy,
    )
    
    model = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )

    class_names = class_names_step(data_folder_path=data_folder_path)


    publish_result = publish_accepted_model(
        model=model,
        approved=approved,
        class_names=class_names,
    )

    return publish_result
