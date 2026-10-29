from pathlib import Path
from steps.training_step import training_fuction
from steps.eval_step import eval_function
from steps.ingestion_step import ingestion_func
from zenml import pipeline
import torch
from typing import Literal



PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_STORE_DIR = PROJECT_ROOT / "data" / "food-101"

RAW_DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_FOLDER_PATH = RAW_DATA_STORE_DIR / 'images'



@pipeline(enable_cache=False, name="food101_training_eval")
def run_training_eval_pipe(
                            
                            freeze,
                            pretrained,
                            
                            optim_name: Literal['Adam', 'Adam_w'],

                            model_name:str, 
                            infer_proc_unit: str,

                            training_proc_unit,

                            data_size=1,
                            data_folder_path=IMAGE_FOLDER_PATH,
                            batch_size=32,
                            train_size=0.7,
                            val_size=0.1,
                            shape_size=(224,224),
                            epochs: int = 3,

                            learning_rate:float= 0.001,
                            loss_label_smoothing: float=0.0,

                            num_classes:int = 101,

                            drop_out:float= 0.2,

                      ):
    

    train_data, val_data, test_data = ingestion_func(
                                        raw_data_store_path=RAW_DATA_STORE_DIR, 
                                        data_size=data_size,
                                        data_folder_path=data_folder_path,
                                        batch_size=batch_size, 
                                        train_size=train_size, 
                                        shape_size=shape_size,
                                        val_size=val_size
                                        )
    
    
    trained_model, model_name = training_fuction( 
                                        training_proc_unit=training_proc_unit, 
                                        model_name=model_name, 
                                        train_data_batch=train_data, 
                                        epochs=epochs,
                                        drop_out=drop_out,
                                        num_classes=num_classes,
                                        optim_name = optim_name,
                                        learning_rate=learning_rate,
                                        loss_label_smoothing = loss_label_smoothing,
                                        val_data_batch=val_data,
                                        freeze=freeze,
                                        pretrained=pretrained

                                     )
    
    _, final_acc, _ = eval_function(model=trained_model,
                                        test_data_batch=test_data, 
                                        infer_proc_unit=infer_proc_unit,
                                        model_name=model_name,
                                        num_classes=num_classes,
                                        )


    return final_acc


'''

if __name__=='__main__':
    run_training_pipe(data_size=0.1,
                      model_name='mobilenet', #custom efficientnet mobilenet
                      proc_unit='xpu', #training stage
                      chip='NPU', #evaluation stage
    )

'''
    
    
