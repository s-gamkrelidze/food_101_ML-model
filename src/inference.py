import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import openvino as ov
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_FOLDER_PATH = PROJECT_ROOT / "best_model"

MODEL_PATH = MODELS_FOLDER_PATH / "active_model.xml"
CLASSES_PATH = MODELS_FOLDER_PATH / "active_classes.json"


@lru_cache(maxsize=4)
def compile_model(
    model_path: str,
    infer_proc_unit: str,
    modified_ns: int,
    file_size: int,
) -> ov.CompiledModel:
    """Compile the active IR once per model version and device."""
    del modified_ns, file_size
    return ov.Core().compile_model(model_path, infer_proc_unit)


def predict_food_image(
    image_base64: str,
    infer_proc_unit:str, #str Literal['CPU', 'NPU', 'GPU'] ='CPU'
) -> dict[str, Any]:
    

    image_bytes = base64.b64decode(image_base64, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        image = source_image.convert("RGB")

    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

    model_stat = MODEL_PATH.stat()
    compiled_model = compile_model(
        str(MODEL_PATH),
        infer_proc_unit,
        model_stat.st_mtime_ns,
        model_stat.st_size,
    )

    image_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    image_tensor = cast(torch.Tensor, image_transform(image)).unsqueeze(0)

    output_layer = compiled_model.output(0)
    array = compiled_model([image_tensor.numpy()])[output_layer]
    logits = torch.from_numpy(array)[0]
    probab = torch.softmax(logits, dim=0)
    top_count = min(5, len(class_names))
    scores, indices = torch.topk(probab, k=top_count)

    top_predictions = [
        {
            "class_name": class_names[int(index.item())],
            "probability": round(float(score.item()),5),
        }
        for score, index in zip(scores, indices)
    ]

    return {
        "predicted_class": top_predictions[0]["class_name"],
        "top_5": top_predictions,
    }


# if __name__=='__main__':
#     print(


#     )
