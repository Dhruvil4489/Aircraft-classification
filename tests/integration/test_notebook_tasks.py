"""Integration tests verifying academic compliance of the completed notebook."""

import json
from pathlib import Path

import pytest

from aircraft_ai import config

COMPLETED_NB_PATH = config.WORKSPACE_ROOT / "notebooks" / "completed" / "Final_Project_Classification_and_Captioning_completed.ipynb"
ORIGINAL_NB_PATH = config.WORKSPACE_ROOT / "notebooks" / "original" / "Final_Project_Classification_and_Captioning.ipynb"


@pytest.fixture
def completed_notebook():
    """Load completed notebook JSON."""
    if not COMPLETED_NB_PATH.exists():
        pytest.skip("Completed notebook has not been generated yet.")
    with open(COMPLETED_NB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.fast
def test_original_notebook_unmodified():
    """Verify original notebook exists and is byte-identical."""
    assert ORIGINAL_NB_PATH.exists(), "Original notebook must exist in notebooks/original/"
    # Check SHA-256 against audit.md record
    from aircraft_ai.data import calculate_file_sha256
    current_hash = calculate_file_sha256(ORIGINAL_NB_PATH)
    expected_hash = "FF61DABC2DC3477C8A69B85815F9D25A695515D31C0B0C774603CC59B86FD7AC".lower()
    assert current_hash.lower() == expected_hash, "Original notebook was modified!"


@pytest.mark.fast
def test_all_10_tasks_present_and_configured(completed_notebook):
    """Verify all 10 tasks in the completed notebook contain the required academic code."""
    cells = completed_notebook["cells"]
    code_sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    all_code = "\n".join(code_sources)

    # Config cell: batch_size = 32 and n_epochs = 5
    assert "batch_size = 32" in all_code or "batch_size=32" in all_code
    assert "n_epochs = 5" in all_code or "n_epochs=5" in all_code

    # Task 1: valid_generator
    assert "valid_datagen.flow_from_directory" in all_code
    assert "directory=valid_dir" in all_code or "valid_dir" in all_code
    assert "shuffle=False" in all_code

    # Task 2: test_generator
    assert "test_datagen.flow_from_directory" in all_code
    assert "directory=test_dir" in all_code or "test_dir" in all_code

    # Task 3: VGG16
    assert "VGG16(weights='imagenet'" in all_code or 'VGG16(weights="imagenet"' in all_code
    assert "include_top=False" in all_code

    # Task 4: compile
    assert "binary_crossentropy" in all_code
    assert "0.0001" in all_code
    assert "accuracy" in all_code

    # Task 5: fit
    assert "model.fit" in all_code
    assert "train_generator" in all_code
    assert "valid_generator" in all_code

    # Task 6: Accuracy plot
    assert "Accuracy Curve" in all_code
    assert "Training Accuracy" in all_code
    assert "Validation Accuracy" in all_code

    # Task 7: visualize prediction
    assert "test_model_on_image" in all_code
    assert "index_to_plot=1" in all_code or "index_to_plot = 1" in all_code

    # Task 8: generate_text helper
    assert "def generate_text" in all_code
    assert "BlipCaptionSummaryLayer" in all_code

    # Task 9: Caption & Task 10: Summary
    assert "generate_text" in all_code
    assert "149_22" in all_code
    assert "summary" in all_code

    # Compatibility: /content/sample_image.jpg must NOT be used
    assert "/content/sample_image.jpg" not in all_code, "Colab-specific path /content/sample_image.jpg must be replaced!"
