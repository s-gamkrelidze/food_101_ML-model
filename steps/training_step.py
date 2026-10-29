
from tqdm import tqdm
from src.params import load_model_for, loss_function, optim_function, time_it
from zenml import step
import mlflow
from pathlib import Path
import torch
from typing import Tuple, Any
import mlflow
from src.params import metrics

SAVE_MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models"


@step(experiment_tracker=True)
@time_it
def training_fuction(
                        model_name,
                        training_proc_unit: str,
                        train_data_batch,
                        val_data_batch,
                        optim_name,
                        learning_rate: float = 0.001,
                        loss_label_smoothing: float = 0.0,
                        drop_out: float = 0.2,
                        num_classes: int = 101,
                        epochs: int = 3,
                        freeze=True,
                        pretrained=True,
                    ) -> Tuple[Any, str]:


    mlflow.log_params({
        "model_name": model_name,
        "training_proc_unit": training_proc_unit,
        "optim_name": optim_name,
        "learning_rate": learning_rate,
        "loss_label_smoothing": loss_label_smoothing,
        "dropout": drop_out,
        "num_classes": num_classes,
        "epochs": epochs,
        "freeze": freeze,
        "pretrained": pretrained,
    })



    device = torch.device(training_proc_unit)
    model_class = load_model_for(model_name=model_name)

    if model_name=='custom':
        model = model_class(num_classes=num_classes, 
                                    dropout=drop_out, 
                                    ).to(device)
    else:    
        model = model_class(num_classes=num_classes, 
                            dropout=drop_out, 
                            freeze=freeze,
                            pretrained=pretrained,
                            ).to(device)

    loss_fn = loss_function(
        loss_name="cross_entropy_normal",
        loss_label_smoothing=loss_label_smoothing,
    )


    trainable = [p for p in model.parameters() if p.requires_grad]

    print("Trainable parameter tensors:", len(trainable))

    optimizer = optim_function(
        optim_learning_rate=learning_rate,
        optim_name=optim_name,
        model_param=trainable,
    )

    # from torch.optim.lr_scheduler import StepLR

    # optim_scheduler= StepLR(optimizer, step_size=3, gamma=0.2)
    #opmtim_scheduler.step()


    train_acc = metrics("acc", num_class=num_classes).to(device)
    val_acc = metrics("acc", num_class=num_classes).to(device)




    print(f"Training started on {device}")

    for epoch in range(epochs):
        model.train()
        train_acc.reset()
        train_loss_sum = 0.0
        train_total = 0


        for batch, (X_train, y_train) in enumerate(tqdm(train_data_batch, desc=f"Epoch {epoch + 1}")):

            X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            y_train_pred = model(X_train)
            loss_train = loss_fn(y_train_pred, y_train)
            loss_train.backward()
            optimizer.step()

            train_acc.update(y_train_pred, y_train)
            train_loss_sum += loss_train.item() * X_train.size(0)
            train_total += y_train.size(0)

            if batch % 500 == 0 and batch != 0:
                print(f"Processed {batch * train_data_batch.batch_size} samples; \n",
                    f"loss={loss_train.item():.4f}"
                )


        train_loss = train_loss_sum / train_total
        train_accuracy = train_acc.compute().item()

        mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch + 1)



        print("-- Validation started")
        model.eval()
        val_acc.reset()
        val_loss_sum = 0.0
        val_total = 0

        with torch.inference_mode():
            for X_val, y_val in val_data_batch:
                X_val, y_val = X_val.to(device), y_val.to(device)

                outputs = model(X_val)
                loss_val = loss_fn(outputs, y_val)

                val_loss_sum += loss_val.item() * X_val.size(0)
                val_total += y_val.size(0)
                val_acc.update(outputs, y_val)


        avg_val_loss = val_loss_sum / val_total
        val_accuracy = val_acc.compute().item()

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.2%}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.2%}"
        )

        mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch + 1)

    model.to("cpu")

    return model, str(model_name)
