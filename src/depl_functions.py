from zenml import step
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS_PATH = PROJECT_ROOT / "saved_models"


# @step(enable_cache=True)
def class_names_step(data_folder_path: str) -> list[str]:
    """Return ImageFolder's alphabetical class order for inference labels."""
    folder = Path(data_folder_path)
    return sorted(path.name for path in folder.iterdir() if path.is_dir())


# @step(enable_cache=True)
def get_last_model_path(model_path: str | Path) -> Path:
    
    folder = Path(model_path)
    model_files = list(folder.glob("*.pth"))

    if not model_files:
        raise FileNotFoundError(f"No .pth files found in {folder}")

    return max(model_files, key=lambda file_path: file_path.stat().st_mtime)
        
    
    """higheest scorer"""
    # folder = Path(model_path)
    # scored_models = []
    # for file_path in folder.glob("*.pth"):
    #     try:
    #         score = float(file_path.stem.rsplit("_acc_", 1)[1])
    #         scored_models.append((score, file_path))
    # return max(scored_models, key=lambda item: item[0])[1]



# @step(enable_cache=True)
def deployment_trigger(model_name: str, min_accuracy: float = 0.70) -> bool:

    checkpoint_path = get_last_model_path(model_path=str(SAVED_MODELS_PATH / model_name)
                                          )
    try:
        final_acc = float(checkpoint_path.stem.rsplit("_acc_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint filename has no valid accuracy score: {checkpoint_path.name}"
        ) from exc

    approved = float(final_acc) >= min_accuracy
    print(
        f"Deployment accuracy={float(final_acc):.4f}, "
        f"required={min_accuracy:.4f}, approved={approved}"
    )

    return approved