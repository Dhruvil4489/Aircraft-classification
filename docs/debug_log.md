# Debug Log & Root Cause Analysis

**Project:** Aircraft Damage Classification, Image Captioning and Summarization  
**Workspace:** `D:\Aircraft classification`  
**Date:** 2026-10-17  

---

## Issue 1: Host Machine Default Python Version Incompatibility
- **Symptom:** Running `py -0p` and `python --version` revealed default host Python was version **3.14.4**.
- **Impact:** Python 3.14 has no pre-compiled binary wheels for `tensorflow-cpu==2.17.1`, `torch==2.2.0`, or `numpy==1.26.4`.
- **Root Cause:** Host system had Python 3.14 registered as the primary launcher default.
- **Resolution:**
  1. Queried winget for Python 3.11: `winget search Python.Python.3.11`.
  2. Installed Python 3.11.9: `winget install --id Python.Python.3.11 -e --silent`.
  3. Created virtual environment explicitly with Python 3.11.9: `py -3.11 -m venv .venv`.
  4. Validated `.venv\Scripts\python.exe` is Python 3.11.9 64-bit.
- **Verification:** `.\.venv\Scripts\python.exe --version` outputs `Python 3.11.9`.

---

## Issue 2: Dependency Collision: NumPy 2.x vs TensorFlow 2.17.1
- **Symptom:** `torchvision==0.17.0` initially downloaded `numpy-2.4.6`.
- **Impact:** TensorFlow 2.17.1 binary builds strictly require `numpy<2` (specifically `numpy==1.26.4`). Running with NumPy 2.x results in runtime `_ARRAY_API` C-extension import crashes.
- **Root Cause:** Package manager resolution defaults to latest major NumPy version when installing PyTorch wheels without pinned NumPy constraints.
- **Resolution:**
  1. Explicitly pinned `numpy==1.26.4` in `requirements.txt`.
  2. Installed pinned version, allowing pip to replace `numpy 2.4.6` with `1.26.4`.
  3. Ran `pip check` to ensure no broken dependencies.
- **Verification:** `pip check` reported *"No broken requirements found."*

---

## Issue 3: Hardcoded Colab Path in Notebook Task 9 Solution Cell
- **Symptom:** In Cell `9ec0f0d9-f86d-41ad-9a24-be03a62f8fa5` of the original notebook, the pre-filled code sets:
  ```python
  image_path = "/content/sample_image.jpg"
  ```
- **Impact:** This file only exists on Google Colab virtual disks. Running locally causes `FileNotFoundError: /content/sample_image.jpg`.
- **Root Cause:** Upstream course notebook author hardcoded a Colab demo file path.
- **Resolution:**
  1. Kept the original notebook byte-identical and unmodified under `notebooks/original/`.
  2. In the completed copy (`notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb`), rebound `image_path` to the test dataset image specified in the preceding markdown instructions:
     ```python
     image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")
     ```
  3. Documented this compatibility change in a dedicated markdown cell.
- **Verification:** Unit test `test_all_10_tasks_present_and_configured` validates that the Colab path is eliminated.

---

## Issue 4: Test Evaluation Step Truncation (`50 // 32 = 1`)
- **Symptom:** Notebook evaluation formula `steps = test_generator.samples // test_generator.batch_size` evaluates only 1 batch (32 samples) and omits 18 test images.
- **Impact:** Evaluating 32 out of 50 samples discards 36% of the test split and produces higher variance.
- **Root Cause:** Integer division `//` in Keras `model.evaluate` does not handle fractional batches unless `math.ceil` is used or full predict arrays are gathered.
- **Resolution:**
  1. Kept the exact notebook formula for academic grading fidelity (reported as 68.75% on 32 samples).
  2. Implemented `full_supplementary_evaluation()` which evaluates all 50 samples across all batches, computing confusion matrix and per-class precision/recall/F1.
- **Verification:** Both metrics are logged in `artifacts/results/metrics.json` and presented side-by-side in the web UI.

---

## Issue 5: Pytest Marker Registration & Substring Match
- **Symptom:** `test_notebook_tasks.py` failed assertion `assert "/content/sample_image.jpg" not in all_code` because a code comment mentioned `/content/sample_image.jpg`. Additionally, `PytestUnknownMarkWarning` warnings were raised for `@pytest.mark.fast`.
- **Root Cause:**
  1. Substring assertion checked all code text including explanatory comments.
  2. Custom markers `fast`, `slow`, `integration` were not registered in a `pytest.ini` config file.
- **Resolution:**
  1. Reworded the code comment to `"Replace Colab sample image path with actual test dataset image"`.
  2. Created `pytest.ini` registering `fast`, `slow`, and `integration` markers.
- **Verification:** Ran `pytest -m fast -v`: 21 passed, 0 failures, 0 unknown marker warnings.

---

## Issue 6: Windows Asyncio Proactor Event Loop Deadlock in `nbclient`/`tornado`
- **Symptom:** Notebook execution using `nbclient` on Windows hung after kernel initialization with the warning:
  `RuntimeWarning: Proactor event loop does not implement add_reader family of methods required for zmq.`
- **Root Cause:** In Python 3.8+ on Windows, the default asyncio event loop policy is `WindowsProactorEventLoopPolicy`. `tornado` and `pyzmq` require socket polling methods only available in `WindowsSelectorEventLoopPolicy`.
- **Resolution:** Explicitly configured `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` on Windows before launching `NotebookClient`.
- **Verification:** Verified kernel communicates bidirectionally without socket deadlock.

---

## Issue 7: Notebook Output Encoding on Windows PowerShell Host
- **Symptom:** Printing notebook execution output directly via PowerShell console in default CP1252 codepage resulted in `UnicodeEncodeError: 'charmap' codec can't encode characters in position 18-37: character maps to <undefined>`.
- **Root Cause:** TensorFlow / Keras 3 training progress bars contain Unicode box drawing and block characters (e.g. `━`). Windows console defaults to legacy CP1252 instead of UTF-8.
- **Resolution:** Set `$env:PYTHONIOENCODING="utf-8"` and ensured all log redirection writes explicit UTF-8 encodings.
- **Verification:** Successfully parsed, printed, and saved notebook execution outputs without encoding errors.

---

## Issue 8: Duplicate BLIP Model Instantiation in Memory Causing Access Violation
- **Symptom:** When running the full test suite (`pytest -v`), `test_blip_layer_and_generate_text` failed with `Windows fatal exception: access violation` inside `torch.nn.modules.linear.py` during `BlipForConditionalGeneration.from_pretrained`.
- **Root Cause:** Both `captioning.py` and `blip_layer.py` maintained separate module-level singleton caches. During integration testing, `captioning.py` initialized one copy of the ~990 MB BLIP weights in memory, and then `blip_layer.py` attempted to instantiate a second distinct copy of the weights into memory while TensorFlow's VGG16 model was also resident in RAM, triggering Windows memory/OpenMP heap allocation corruption.
- **Resolution:** Refactored `blip_layer.get_blip_components()` to delegate directly to `captioning.get_blip_model()`, strictly enforcing a single shared singleton instance across the entire application.
- **Verification:** Ran the full test suite (`pytest tests/unit/ tests/integration/ -v`). All 24 tests passed cleanly in 27.60s without memory errors or access violations.

