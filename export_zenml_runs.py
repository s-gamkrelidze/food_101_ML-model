"""Export this project's ZenML training runs and evaluation outputs to CSV."""

import csv
from pathlib import Path
from typing import Any

from zenml.client import Client


PIPELINE_NAME = "food101_training_eval"
OUTPUT_FILE = Path(__file__).resolve().parent / "zenml_model_comparison.csv"


def scalar(value: Any) -> Any:
    """Convert scalar tensor-like values to ordinary Python values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


client = Client()
page = client.list_pipeline_runs(
    pipeline_name=PIPELINE_NAME,
    sort_by="asc:created",
    page=1,
    size=100,
    hydrate=True,
)
pipeline_runs = list(page.items)
for page_number in range(2, page.total_pages + 1):
    next_page = client.list_pipeline_runs(
        pipeline_name=PIPELINE_NAME,
        sort_by="asc:created",
        page=page_number,
        size=100,
        hydrate=True,
    )
    pipeline_runs.extend(next_page.items)

rows = []
for run in pipeline_runs:
    row = {
        "run_name": run.name,
        "run_id": str(run.id),
        "status": str(run.status),
        "created": str(run.created),
    }

    # Pipeline parameters include such values as epochs, batch_size, and
    # learning_rate. Training step parameters include model_name and optimizer.
    pipeline_params = getattr(run.config, "parameters", None) or {}
    row.update(pipeline_params)

    steps = run.steps
    training_step = steps.get("training_fuction")
    if training_step is not None:
        row.update(getattr(training_step.config, "parameters", None) or {})
        training_start = training_step.start_time
        training_end = training_step.end_time
        row["processing_time_seconds"] = (
            round((training_end - training_start).total_seconds(), 2)
            if training_start is not None and training_end is not None
            else None
        )

    # Evaluation returns named ZenML artifacts: avg_test_loss, final_acc,
    # and final_f1. Load those scalar artifacts for the comparison table.
    eval_step = steps.get("eval_function")
    if eval_step is not None:
        for output_name, artifacts in eval_step.outputs.items():
            if not artifacts:
                continue
            try:
                row[output_name] = scalar(artifacts[0].load())
            except Exception as exc:
                row[output_name] = f"Could not load artifact: {exc}"

    rows.append(row)

if not rows:
    print(f"No runs found for pipeline {PIPELINE_NAME!r}.")
else:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} runs to {OUTPUT_FILE}")
