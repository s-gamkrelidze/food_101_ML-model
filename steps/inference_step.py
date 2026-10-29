from typing import Any
import click
from zenml import step, log_metadata

from src.inference import predict_food_image
from pprint import pformat



@step(enable_cache=False)
def predict_image_step(image_base64: str, infer_proc_unit: str) -> dict[str, Any]:
    result = predict_food_image(image_base64, infer_proc_unit)
    
    click.echo(pformat(result))
        
        
    log_metadata(metadata={
        "inference_device": infer_proc_unit,
        "prediction_result": result,
    })



    return result