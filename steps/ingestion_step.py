from zenml import step
from src.ingestion_functions import ingest, batch_tranform, data_downloader
from typing import Tuple, Annotated, Any
from pathlib import Path



@step()
def ingestion_func(
                   raw_data_store_path, 
                   data_folder_path,
                   data_size, 
                   batch_size=32, 
                   train_size:float=0.8, 
                   shape_size=(224,224),
                   val_size:float=0.1,

                   ) -> Tuple[Annotated[Any, "train_data"], Annotated[Any, "val_data"], Annotated[Any, "test_data"], ]:
    
    data_downloader(raw_data_store_path=raw_data_store_path, download=False)

    dataset=ingest(data_folder_path=data_folder_path, 
                   shape_size=shape_size)
    
    train_data_batch, val_data_batch, test_data_batch = batch_tranform(dataset, 
                                                       data_size, 
                                                       batch_size=batch_size, 
                                                       train_size=train_size,
                                                       val_size=val_size)
    
    return train_data_batch, val_data_batch, test_data_batch


