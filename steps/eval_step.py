import openvino as ov
import torch
from typing import Literal
from zenml import step
from torch import nn
from pathlib import Path
from src.params import metrics
from typing import Annotated, Tuple, Any
from datetime import datetime
from tqdm import tqdm


MODEL_SAVE_PATH=Path(__file__).resolve().parent.parent / 'saved_models'


@step(experiment_tracker=True)
def eval_function(model,
                  model_name,
                  test_data_batch,
                  num_classes,
                  infer_proc_unit, #['NPU', 'GPU', 'CPU'],
                  ) -> Tuple[
    Annotated[float, "avg_test_loss"],
    Annotated[float, "final_acc"],
    Annotated[float, "final_f1"],
]:


    acc = metrics('acc', num_class=num_classes)
    f1_score = metrics('f1_score', num_class=num_classes)
    loss_fn = nn.CrossEntropyLoss()


    print(f'Evaluation started on --{infer_proc_unit}-- processing unit.')



    model.eval()

    sample_batch_X, _ = next(iter(test_data_batch))

    ov_model = ov.convert_model(model, example_input=sample_batch_X, dynamo=True)
    comp_model = ov.compile_model(ov_model, device_name=str(infer_proc_unit))
    output_layer = comp_model.output(0)
    
    test_loss = 0.0
    num_batches = 0
    
    acc.reset()
    f1_score.reset()
    
    with torch.inference_mode():
        for X_test, y_test in test_data_batch:
            X_test_np = X_test.cpu().numpy()
            
            ov_output = comp_model([X_test_np])[output_layer]
            logits = torch.from_numpy(ov_output)
            
            test_loss += loss_fn(logits, y_test).item()
            num_batches += 1
            
            y_test_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            acc.update(y_test_pred, y_test)
            f1_score.update(y_test_pred, y_test)

    final_acc = acc.compute()
    final_f1 = f1_score.compute()
    avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_filename = f"{model_name}_{timestamp}_acc_{final_acc.item():.4f}.pth"        


    save_path = MODEL_SAVE_PATH / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    save_mode_path = save_path / model_filename

    torch.save(
        {name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()},
        save_mode_path,
    )


    print(f"Results -> Loss: {avg_test_loss:.4f}, Acc: {final_acc:.4f}, F1: {final_f1:.4f}")
    print(f"Model saved successfully to {save_mode_path}/n/n")

    return float(avg_test_loss), float(final_acc), float(final_f1)



