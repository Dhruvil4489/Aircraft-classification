# Project Completion Report — Aircraft Damage Classification and Multimodal Captioning

**Project Title:** Aircraft Structural Damage Classification (VGG16) & Multimodal Captioning/Summarization (BLIP)  
**Workspace:** `D:\Aircraft classification`  
**Execution Platform:** Windows 11 Home Single Language (AMD64), Python 3.11.9 Virtual Environment (`.venv`)  
**Date of Completion:** 2026-10-17  
**Status:** All Acceptance Criteria Implemented and Verified (Zero Mock Data, Real Execution)

---

## 1. Acceptance Checklist (Section R)

Every requirement specified in Section R has been implemented and strictly verified against real execution artifacts.

| # | Requirement Item | Status | Verification Evidence / File Path |
|---|------------------|:------:|-----------------------------------|
| **R01** | Original notebook inspected and left unmodified (SHA-256 matches) | **Implemented & Verified** | `notebooks/original/Final_Project_Classification_and_Captioning.ipynb`<br>SHA-256: `FF61DABC2DC3477C8A69B85815F9D25A695515D31C0B0C774603CC59B86FD7AC` |
| **R02** | Every explicit notebook requirement accounted for in audit document | **Implemented & Verified** | [audit.md](file:///D:/Aircraft%20classification/docs/audit.md) (Checklist items R01–R31) |
| **R03** | All 10 academic tasks implemented with exact parameters | **Implemented & Verified** | [Final_Project_Classification_and_Captioning_completed.ipynb](file:///D:/Aircraft%20classification/notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb) |
| **R04** | Dataset structure, counts, and class mapping verified (`crack`: 0, `dent`: 1) | **Implemented & Verified** | [dataset_report.json](file:///D:/Aircraft%20classification/artifacts/results/dataset_report.json)<br>Train: 300, Valid: 96, Test: 50. Zero corrupt images. |
| **R05** | VGG16 loaded with ImageNet weights, `include_top=False`, frozen backbone, specified head | **Implemented & Verified** | [classifier.py](file:///D:/Aircraft%20classification/src/aircraft_ai/classifier.py)<br>[training.log](file:///D:/Aircraft%20classification/artifacts/logs/training.log) (27,823,425 params; 13,108,737 trainable) |
| **R06** | Model trained 5 epochs on `train_generator` with `valid_generator` | **Implemented & Verified** | [history.json](file:///D:/Aircraft%20classification/artifacts/results/history.json)<br>Final Epoch: Train Acc 88.00%, Val Acc 70.83% |
| **R07** | Loss and accuracy curves generated from real history | **Implemented & Verified** | [training_loss.png](file:///D:/Aircraft%20classification/artifacts/results/training_loss.png), [validation_loss.png](file:///D:/Aircraft%20classification/artifacts/results/validation_loss.png), [accuracy_curve.png](file:///D:/Aircraft%20classification/artifacts/results/accuracy_curve.png) |
| **R08** | Evaluated on test split (notebook formula + full supplementary evaluation) | **Implemented & Verified** | [metrics.json](file:///D:/Aircraft%20classification/artifacts/results/metrics.json)<br>Notebook formula: Acc 68.75% (32 samples); Full test: Acc 76.00% (50 samples) |
| **R09** | Real test predictions visualized (Task 7 and deterministic grid) | **Implemented & Verified** | [task7_prediction.png](file:///D:/Aircraft%20classification/artifacts/results/task7_prediction.png), [test_predictions_grid.png](file:///D:/Aircraft%20classification/artifacts/results/test_predictions_grid.png) |
| **R10** | Model saved, reloaded, predictions match | **Implemented & Verified** | [vgg16_aircraft_damage.keras](file:///D:/Aircraft%20classification/artifacts/models/vgg16_aircraft_damage.keras)<br>Max prediction diff: $0.00 \times 10^0$ (atol $10^{-5}$) |
| **R11** | New image classified via real inference with identical preprocessing (224x224, 1/255) | **Implemented & Verified** | [inference.py](file:///D:/Aircraft%20classification/src/aircraft_ai/inference.py)<br>[api.log](file:///D:/Aircraft%20classification/artifacts/logs/api.log) (P(dent)=0.9952 for Task 9 image) |
| **R12** | BLIP loads from `"Salesforce/blip-image-captioning-base"` | **Implemented & Verified** | [captioning.py](file:///D:/Aircraft%20classification/src/aircraft_ai/captioning.py)<br>[blip_execution.log](file:///D:/Aircraft%20classification/artifacts/logs/blip_execution.log) |
| **R13** | Real caption and real summary generated for Task 9/10 image | **Implemented & Verified** | [blip_examples.json](file:///D:/Aircraft%20classification/artifacts/results/blip_examples.json) |
| **R14** | `BlipCaptionSummaryLayer` and `generate_text` implemented and tested on real image | **Implemented & Verified** | [blip_layer.py](file:///D:/Aircraft%20classification/src/aircraft_ai/blip_layer.py)<br>[test_end_to_end.py](file:///D:/Aircraft%20classification/tests/integration/test_end_to_end.py) |
| **R15** | Errors and missing dependencies produce clear diagnostics | **Implemented & Verified** | [errors.py](file:///D:/Aircraft%20classification/src/aircraft_ai/errors.py)<br>[test_api.py](file:///D:/Aircraft%20classification/tests/unit/test_api.py) (HTTP 400, 413, 503) |
| **R16** | Completed notebook contains real executed outputs for all 10 tasks | **Implemented & Verified** | [Final_Project_Classification_and_Captioning_completed.ipynb](file:///D:/Aircraft%20classification/notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb)<br>[notebook_execution.log](file:///D:/Aircraft%20classification/artifacts/logs/notebook_execution.log) |
| **R17** | App starts with documented commands; every UI control works | **Implemented & Verified** | [api.py](file:///D:/Aircraft%20classification/src/aircraft_ai/api.py), [frontend/index.html](file:///D:/Aircraft%20classification/frontend/index.html)<br>Browser subagent verified: `web_ui_smoke_test_1792244144524.webp` |
| **R18** | README matches implementation | **Implemented & Verified** | [README.md](file:///D:/Aircraft%20classification/README.md) |
| **R19** | Tests run; results recorded and categorised | **Implemented & Verified** | [test_run.log](file:///D:/Aircraft%20classification/artifacts/logs/test_run.log)<br>24 passed, 0 failed, 0 skipped in 27.60s |
| **R20** | No core functionality uses hardcoded/demo results | **Implemented & Verified** | Zero mocks, zero fake data, all numbers tied to real artifact files |
| **R21** | Limitations disclosed in all interfaces and documentation | **Implemented & Verified** | Disclaimed in README, report outline, API response payloads, and Web UI |

---

## 2. Execution Environment & Pinned Versions

The exact software and hardware environment used to train, evaluate, and test the system:

```text
Host OS: Windows 11 Home Single Language (Version 10.0.26100 AMD64)
Python Version: 3.11.9 (v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
Virtual Environment: D:\Aircraft classification\.venv

Key Package Versions:
- tensorflow-cpu: 2.17.1
- keras: 3.15.1
- torch (CPU wheel): 2.2.0+cpu
- torchvision (CPU wheel): 0.17.0+cpu
- transformers: 4.38.2
- pillow: 11.1.0
- pandas: 2.2.3
- matplotlib: 3.9.2
- numpy: 1.26.4 (pinned <2.0 for TF 2.17 compatibility)
- fastapi: 0.135.1
- uvicorn: 0.41.0
- scikit-learn: 1.6.1
- pytest: 9.1.1
- nbclient: 0.10.4
- nbformat: 5.10.4
```

---

## 3. Actual Metrics

### 3.1 Training Progression (`history.json`)
Training parameters: `seed_value=42`, `batch_size=32`, `n_epochs=5`, `optimizer=Adam(lr=0.0001)`, `loss=binary_crossentropy`.

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|:-----:|:-------------:|:-----------------:|:---------------:|:-------------------:|
| **1** | 0.7201 | 53.67% (161/300) | 0.6338 | 60.42% (58/96) |
| **2** | 0.5535 | 72.33% (217/300) | 0.5821 | 69.79% (67/96) |
| **3** | 0.4737 | 77.67% (233/300) | 0.5486 | 71.88% (69/96) |
| **4** | 0.4123 | 81.33% (244/300) | 0.6457 | 65.62% (63/96) |
| **5** | **0.3204** | **88.00%** (264/300) | **0.5098** | **70.83%** (68/96) |

### 3.2 Evaluation on Test Split (`metrics.json`)

#### A. Academic Notebook Truncated Formula (`steps = samples // batch_size`)
- Formula evaluated: `steps = 50 // 32 = 1` step
- Evaluated samples: **32** (18 samples omitted due to integer truncation)
- Test Loss: **0.7326**
- Test Accuracy: **68.75%** (22 / 32 correct)

#### B. Full Supplementary Evaluation (Complete Test Split)
- Evaluated samples: **50** / 50 (100% complete evaluation)
- Test Loss: **0.5783**
- Test Accuracy: **76.00%** (38 / 50 correct)
- **Confusion Matrix:**
  - True Negative (`crack`): **15**
  - False Positive (`dent`): **10**
  - False Negative (`crack`): **2**
  - True Positive (`dent`): **23**
- **Per-Class Metrics:**
  - **Positive Class (`dent`):** Precision = 0.6970 (23/33), **Recall = 0.9200 (23/25)**, **F1 = 0.7931**
  - **Negative Class (`crack`):** Precision = 0.8824 (15/17), Recall = 0.6000 (15/25), F1 = 0.7143
- **Imbalance & Sample Size Discussion:** The test set is perfectly balanced (25 crack, 25 dent). High dent recall (92.0%) indicates the model detects surface depressions with high sensitivity. The primary error mode was crack false positives (10 cracks predicted as dents), where shadow gradients on curved metallic fuselage panels mimic dent depressions.

---

## 4. Actual Caption & Summary Outputs

All captions and summaries were generated by loading real weights from `Salesforce/blip-image-captioning-base` (PyTorch CPU execution) and are persisted in `artifacts/results/blip_examples.json` and inside the completed notebook.

### Task 9 & Task 10 Evaluation Image
- **Image File:** `aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg`
- **Task 9 (Caption, Prompt: `"This is a picture of"`):**
  > `"this is a picture of a plane that was sitting on the ground in a field"`
- **Task 10 (Summary, Prompt: `"This is a detailed photo showing"`):**
  > `"this is a detailed photo showing the damage to the fuselage of the aircraft"`

### Section 2.2 Demonstration Image
- **Image File:** `aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg`
- **Caption (Prompt: `"This is a picture of"`):**
  > `"this is a picture of a plane"`
- **Summary (Prompt: `"This is a detailed photo showing"`):**
  > `"this is a detailed photo showing the engine of a boeing 747"`

---

## 5. Compatibility Changes Made to Notebook

As mandated by Ground Rule 0 and Section K, the original notebook `notebooks/original/Final_Project_Classification_and_Captioning.ipynb` was never modified. All changes were isolated to `notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb` and documented in a "Compatibility changes" markdown cell:

1. **Task 9 Colab Path Fix:** The pre-filled lab notebook contained `image_path = "/content/sample_image.jpg"`, an invalid Google Colab path that overrides the dataset path and causes runtime failure. This was replaced with the actual dataset path: `aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg`.
2. **Task 10 Implementation:** Task 10 was implemented using the same image with task `"summary"`.
3. **Dataset Download Idempotence Guard:** The original lab cell unconditionally deleted and re-downloaded the tarball. In the completed notebook, extraction is guarded to verify existing data integrity without redownloading.
4. **Pre-installed Dependencies:** The `!pip install` bash cells were marked as unexecuted since dependencies are preinstalled in the clean `.venv`.
5. **Keras 3 Subclassing Compatibility:** In `BlipCaptionSummaryLayer`, the `call()` method was adapted to accept both `(image_path, task)` and single tuple `(image_path, task)` inputs, maintaining strict backward compatibility across Keras 2 and Keras 3.
6. **Environment Cell Added:** A final cell was appended printing runtime versions for Python, TensorFlow, Keras, PyTorch, Transformers, Pillow, Pandas, and Matplotlib.

---

## 6. Test Suite Execution Summary

The pytest suite was executed against the active virtual environment and recorded in [test_run.log](file:///D:/Aircraft%20classification/artifacts/logs/test_run.log):

- **Total Tests Collected:** 24
- **Passed:** **24**
- **Failed:** **0**
- **Skipped:** **0**
- **Could-not-run:** **0**
- **Total Execution Time:** 27.60 seconds

### Categorized Breakdown:
- **Unit Tests (19 passed):**
  - `test_api.py` (6 passed): Health check, class mapping, dataset info, validation error handling, payload size limits, missing figure 404s.
  - `test_blip.py` (3 passed): Prompt construction, invalid task exception handling, invalid file error handling.
  - `test_classifier.py` (4 passed): Model parameter counts, frozen backbone verification, binary head shapes, loss/optimizer compilation.
  - `test_data.py` (6 passed): Generator discovery, binary class mapping (`crack`: 0, `dent`: 1), PIL image verification, safe tar member extraction.
- **Integration Tests (5 passed):**
  - `test_end_to_end.py` (3 passed): Real ImageNet VGG16 loading, real Hugging Face BLIP captioning and summarization, custom `BlipCaptionSummaryLayer` and `generate_text` execution returning valid non-sentinel `tf.string` tensors.
  - `test_notebook_tasks.py` (2 passed): Automated validation of completed notebook structure across all 10 academic tasks, verified absence of hardcoded Colab paths.

---

## 7. Known Limitations & Technical Constraints

1. **"Summary" Semantics:** The model used for summarization is identical to the captioning checkpoint (`Salesforce/blip-image-captioning-base`). It generates descriptions conditioned on the prompt prefix `"This is a detailed photo showing"`. It does not perform document summarization and carries no factual safety guarantees.
2. **Aviation Safety Disclaimer:** This system is developed solely for educational and academic demonstrator purposes. It is **not** certified under FAA/EASA airworthiness standards and must never be used as a primary structural safety assessment tool.
3. **Keras Layer Serialization:** The custom `BlipCaptionSummaryLayer` wraps PyTorch inference inside `tf.py_function`. It is not exportable to a standard TensorFlow SavedModel graph or ONNX runtime, and cannot participate in gradient backpropagation.
4. **Small Sample Variance:** The test split contains 50 images. With each sample representing 2.0% of total accuracy, metrics are sensitive to individual misclassifications.
5. **Windows CPU Execution:** TensorFlow on Windows native AMD64 operates CPU-only without GPU/CUDA. Both VGG16 and BLIP run in CPU mode; BLIP inference is guarded by a concurrency lock of 1 to prevent memory exhaustion.

---

## 8. Exact Reproduction Commands

All commands are runnable from Windows PowerShell in the project workspace:

### 8.1 Environment Setup
```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install locked dependencies
python -m pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.lock.txt
pip check
```

### 8.2 Dataset Preparation
```powershell
python scripts/prepare_data.py
```

### 8.3 Training VGG16 Classifier
```powershell
python scripts/run_train.py
```

### 8.4 Evaluation & Plot Generation
```powershell
python scripts/run_eval.py
```

### 8.5 BLIP Captioning & Summarization Examples
```powershell
python scripts/run_blip_examples.py
```

### 8.6 Execute Automated Test Suite
```powershell
python -m pytest -v
```

### 8.7 Execute Completed Notebook End-to-End
```powershell
python scripts/execute_notebook.py
```

### 8.8 Launch Web Application & API Service
```powershell
python -m uvicorn aircraft_ai.api:app --host 127.0.0.1 --port 8000 --app-dir src
# Open http://127.0.0.1:8000 in your browser
```
