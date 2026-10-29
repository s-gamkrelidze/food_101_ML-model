from base64 import b64encode
from pathlib import Path
import click
from pipeline.train_eval_test_pipe import run_training_eval_pipe
from pipeline.inference_pipe import inference_pipe
from typing import Literal
from pipeline.deploy_pipe import run_deployment_pipe
from datetime import datetime
from zoneinfo import ZoneInfo

local_time = datetime.now(ZoneInfo("Asia/Tbilisi")).strftime("%Y_%m_%d_%H_%M_%S_%f")


@click.command()
@click.option(
    "--config",
    type=click.Choice(["train-deploy", "infer"]),
    default="infer",
    show_default=True,
    help="Train/evaluate and publish an accepted model, or predict one local image.",
)

@click.option(
    '--freeze/--no-freeze',
    default=True,
    show_default=True,
    help="Freeze pretrained feature-extraction layers.",
)
@click.option(
    '--pretrained/--no-pretrained',
    default=True,
    show_default=True,
    help="Initialize pretrained model layers.",
)

@click.option(
    "--image-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Path to the image file.",
)

@click.option(
    "--min-accuracy",
    type=float,
    default=0.70,
    show_default=True,
    help="Accuracy required before replacing the active inference checkpoint.",
)
@click.option("--epochs", 
              type=int, 
              default=2, 
              show_default=True
)
@click.option(
    "--training-proc-unit",
    type=click.Choice(["xpu", "cpu"]),
    default="cpu",
    show_default=True,
    help="Training stage proccesing unit types",
)

@click.option(
    "--infer-proc-unit",
    type=click.Choice(["NPU", "GPU", "CPU"]),
    default="CPU",
    show_default=True,
    help="Processing unit used for evaluation and inference mode",
)

@click.option(
    "--model-name",
    "model_name",
    type=click.Choice(["mobilenet", "efficientnet", "custom"]),
    default="efficientnet",
    show_default=True,
    help="Model architecture for training.",
)
@click.option(
    "--data-size",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1,
    show_default=True,
    help="Fraction of the dataset to use (0.1 ->> 10%).",
)

@click.option(
    "--learning-rate",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0001,
    show_default=True,
)


def run_deployment(
    config: str,                
    image_path: Path | None,
    min_accuracy: float,       
    model_name:str,             
    data_size:float,            
    training_proc_unit:str,     
    infer_proc_unit: Literal['CPU', 'NPU', 'GPU'] ='CPU',   
    optim_name: Literal['Adam', 'Adam_w']='Adam_w',
    epochs:int = 2,
    learning_rate:float= 0.0001,
    loss_label_smoothing:float=0.0,
    num_classes:int = 101,
    drop_out:float=0.2,
    freeze: bool = True,
    pretrained: bool =True,




) -> None:
    """Run continuous deployment or local inference."""
    if config == "train-deploy":
        run_training_eval_pipe.with_options(
            run_name=f"train_{model_name}_{local_time}",
            enable_cache=False
        )(
            data_size=data_size,
            model_name=model_name,
            training_proc_unit=training_proc_unit,
            infer_proc_unit=infer_proc_unit,
            epochs=epochs,
            train_size=0.7,
            optim_name = optim_name,
            learning_rate=learning_rate,
            loss_label_smoothing=loss_label_smoothing,
            num_classes=num_classes,
            drop_out=drop_out,
            val_size=0.1,
            freeze=freeze,
            pretrained=pretrained,
            batch_size=32,
            )


        result = run_deployment_pipe.with_options(
            run_name=f"deploy_{model_name}_{local_time}"
        )(
            min_accuracy=min_accuracy,
            model_name=model_name,
        )


        print(
            # click.echo(result,
            'Finished!!',
            result,
            )
        
        
        return


    if image_path is None:
        raise click.UsageError("Pass image path when --config predict is selected.")

    image_base64 = b64encode(image_path.read_bytes()).decode("ascii")
    result = inference_pipe.with_options(
                                    run_name="inference_{date}_{time}"
                                    )(image_base64=image_base64,
                                      infer_proc_unit=infer_proc_unit)

    
    # print(result)


if __name__=='__main__':
    run_deployment()


