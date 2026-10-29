import base64
from io import BytesIO

import gradio as gr
from PIL import Image
import spaces

from src.inference import CLASSES_PATH, MODEL_PATH, predict_food_image


@spaces.GPU
def classify_image(image: Image.Image | None) -> tuple[str, dict[str, float]]:
    """Classify an uploaded image and return its top five predictions."""
    if image is None:
        raise gr.Error("ჯერ ატვირთეთ კერძის სურათი.")

    required_files = (MODEL_PATH, MODEL_PATH.with_suffix(".bin"), CLASSES_PATH)
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise gr.Error(
            "მოდელის ფაილები ვერ მოიძებნა: "
            + ", ".join(missing_files)
            + ". განათავსეთ ისინი best_model/ საქაღალდეში."
        )

    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    result = predict_food_image(image_base64=image_base64, infer_proc_unit="CPU")
    top_predictions = {
        item["class_name"]: float(item["probability"])
        for item in result["top_5"]
    }
    return result["predicted_class"], top_predictions


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="ატვირთეთ საკვების ფოტო"),
    outputs=[
        gr.Textbox(label="პროგნოზირებული კლასი"),
        gr.Label(label="ტოპ 5 პროგნოზი", num_top_classes=5),
    ],
    title="Food-101 — საკვების ამოცნობა",
    description="ატვირთეთ საკვების ფოტო. მოდელი დააბრუნებს ყველაზე სავარაუდო კლასსა და ტოპ 5 შედეგს.",
)


if __name__ == "__main__":
    demo.launch()