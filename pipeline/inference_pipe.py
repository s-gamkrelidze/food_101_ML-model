from typing import Any

from zenml import pipeline
from zenml.deployers.local.local_deployer import LocalDeployerSettings
from steps.inference_step import predict_image_step

settings: dict[str, Any] = {
    "deployer": LocalDeployerSettings(
        port=8000,  #type: ignore
        blocking=True,   #type: ignore
    )
}


@pipeline(
    name="food101_image_inference",
    enable_cache=False,
    settings=settings,
    )

def inference_pipe(infer_proc_unit, image_base64: str = "", ):
    result= predict_image_step(image_base64=image_base64, infer_proc_unit=infer_proc_unit)
    
    return result

