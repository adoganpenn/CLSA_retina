import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "Age_Glaucoma"
    / "11_end_to_end_two_group_retinal_analysis.ipynb"
)


def notebook_text():
    payload = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )


def test_quick_start_and_supported_inputs_are_documented():
    text = notebook_text()
    assert "## Quick start" in text
    for value in ("JPEG", "DICOM", "ZIP", "Delta", "Parquet", "CSV"):
        assert value in text


def test_expensive_stages_are_batched_and_resumable():
    text = notebook_text()
    for stage in ("quality", "embedding", "segmentation", "explainability"):
        assert f"{stage}_checkpointed_incomplete" in text
    assert "max_new_batches_per_stage" in text
    assert "consolidate_batch_parquets" in text
    assert "run_progress.parquet" in text
    assert "run_root / \"_SUCCESS.json\"" in text


def test_privacy_and_widget_contract():
    text = notebook_text()
    assert "credentials_persisted\": False" in text
    assert "dbutils.widgets.set" not in text
    assert "entire matched set remains within one fold" in text
    assert "perfectly confounded" in text
