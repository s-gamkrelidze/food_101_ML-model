import json
import os
from pathlib import Path
from typing import Any, Literal
import torch
from torch import nn
import torch
from zenml import step
import openvino as ov
import numpy as np



MODEL_DIR = Path(__file__).resolve().parents[1] / "best_model"
ACTIVE_XML = MODEL_DIR / "active_model.xml"
ACTIVE_BIN = MODEL_DIR / "active_model.bin"
ACTIVE_CLASSES = MODEL_DIR / "active_classes.json"
EXAMPLE_TEST_DATA_FOR_OPENVINO=Path(__file__).resolve().parent.parent / "data" /"openvino_example_input.npy"


@step(enable_cache=False)
def publish_accepted_model(
    model: nn.Module,
    approved: bool,
    class_names: list[str],
) -> str:


    if not approved:
        return (
            "Model was rejected by the accuracy gate fro not beeing accurate enought. "
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)


    example_input = np.load(EXAMPLE_TEST_DATA_FOR_OPENVINO)

    model = model.to('cpu').eval()
    ov_model = ov.convert_model(
        model,
        example_input=example_input[:1],
        dynamo=True,
    )

    # The temporary XML and BIN share a basename.
    tmp_xml = MODEL_DIR / "active_model.tmp.xml"
    tmp_bin = MODEL_DIR / "active_model.tmp.bin"
    tmp_classes = MODEL_DIR / "active_classes.tmp.json"

    try:

        ov.save_model(ov_model, str(tmp_xml), compress_to_fp16=False)

        tmp_classes.write_text(
            json.dumps(class_names, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Replace the BIN first and XML last; XML is the entry point.
        os.replace(tmp_bin, ACTIVE_BIN)
        os.replace(tmp_classes, ACTIVE_CLASSES)
        os.replace(tmp_xml, ACTIVE_XML)
    finally:
        tmp_xml.unlink(missing_ok=True)
        tmp_bin.unlink(missing_ok=True)
        tmp_classes.unlink(missing_ok=True)

    return f"Published OpenVINO model to {ACTIVE_XML}"
