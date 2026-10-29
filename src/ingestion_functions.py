from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from torch import Generator

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
import random


def data_downloader(raw_data_store_path, download=False):
    if download == True:
        print(f"Dataset downloading has been started to {raw_data_store_path}")
        datasets.Food101(
            root=raw_data_store_path,
            split='train',
            transform=ToTensor(),
            download=True,
        )


    # DATA_PATH = Path(__file__).resolve().parent.parent / "data" / 'food101' / 'images'

    # SEED=42
    # APPLY=True

    # rng=random.Random(SEED)
    # classes=sorted(p for p in DATA_PATH.iterdir() if p.is_dir())
    # total_before=total_after=0

    # if len(classes)!=101:
    #     raise RuntimeError(f"Expected 101 classes, found {len(classes)}")

    # for folder in classes:
    #     images=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"})
    #     rng.shuffle(images)
    #     keep=(len(images)+1)//2
    #     remove=images[keep:]
    #     total_before+=len(images)
    #     total_after+=keep

    #     if APPLY:
    #         for image in remove:
    #             image.unlink()

    #     print(f"{folder.name}: {len(images)} -> {keep}")

    # print(f"\nClasses: {len(classes)}")
    # print(f"Images before: {total_before:,}")
    # print(f"Images remaining: {total_after:,}")
    # print(f"Images {'deleted' if APPLY else 'to delete'}: {total_before-total_after:,}")
    # print("DRY RUN: no files deleted." if not APPLY else "Deletion complete.")
    
    return None



def ingest(data_folder_path, shape_size=(224,224)):

    transf=transforms.Compose([
        transforms.Resize(shape_size),
        transforms.ToTensor(),
    ])

    dataset=datasets.ImageFolder(
        root=data_folder_path,
        transform=transf,
    )

    return dataset



def batch_tranform(dataset, data_size:float, batch_size=64, train_size=0.7, val_size=0.1, seed=42):

    generator = Generator().manual_seed(seed)


    half_dataset = int(len(dataset)*data_size)

    train_size=int(train_size*half_dataset)
    val_size=int(val_size*half_dataset)
    test_size = half_dataset - train_size - val_size

    reduced_dataset = Subset(dataset, range(half_dataset), )



    train_data, val_data, test_data = random_split(reduced_dataset, [train_size, val_size, test_size], generator=generator)

    train_data_batch = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )   

    val_data_batch = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    test_data_batch = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    img, label = next(iter(test_data_batch))

    SAMPLE_TEST_IMG_DATA=Path(__file__).resolve().parent.parent
    

    np.save(SAMPLE_TEST_IMG_DATA / "data" /"openvino_example_input.npy", img[:1].cpu().numpy())
    print(f'Samlple test_dataset example was saved for openvino to')



    print(f'Train data_batch len: {len(train_data_batch)},\n Validation data_batch: {len(val_data_batch)},\n len Test batch len: {len(test_data_batch)}')
    print(f"Batch data dimension {img.shape}, Batch label shape {label.shape}")

    return train_data_batch, val_data_batch, test_data_batch


