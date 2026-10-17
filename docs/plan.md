# Architecture and Implementation Plan

**Project:** Aircraft Damage Classification, Image Captioning and Summarization  
**Workspace:** `D:\Aircraft classification`  
**Plan Date:** 2026-10-17  

---

## 1. Architectural Overview and Principles

```
                       +---------------------------------------+
                       |              Browser UI               |
                       | (frontend: HTML, CSS, JavaScript ES6)  |
                       +-------------------+-------------------+
                                           | HTTP / REST
                                           v
                       +---------------------------------------+
                       |            FastAPI Service            |
                       |       (src/aircraft_ai/api.py)        |
                       +---------+-------------------+---------+
                                 |                   |
                     Lazy Loaded |                   | Lazy Loaded
                     Singleton   |                   | Singleton (Lock=1)
                                 v                   v
      +--------------------------------+   +--------------------------------+
      |       TensorFlow / Keras       |   |       PyTorch / Transformers   |
      |             VGG16              |   |              BLIP              |
      |   - Frozen Feature Extractor   |   |   - Processor & Conditional    |
      |   - 512-512-1 Binary Head      |   |     Generation Base Model      |
      |   - Input: (224, 224, 3) 1/255 |   |   - Caption / Summary Prompt   |
      +--------------------------------+   +--------------------------------+
                                 \                   /
                                  \                 /
                                   v               v
                   +-----------------------------------------------+
                   |              Framework Boundary               |
                   | Only strings (file paths in, text/JSON out)   |
                   | cross between TensorFlow and PyTorch domains. |
                   +-----------------------------------------------+
```

### Key Tenets
1. **Academic Fidelity (P0 First):** All 10 tasks in the notebook must match original parameters exactly:
   - `batch_size = 32`, `n_epochs = 5`, `seed_value = 42`.
   - VGG16 with `weights='imagenet'`, `include_top=False`, `Flatten()`, freeze all layers.
   - Head: `Dense(512, relu)`, `Dropout(0.3)`, `Dense(512, relu)`, `Dropout(0.3)`, `Dense(1, sigmoid)`.
   - Adam optimizer with `learning_rate=0.0001`, `loss='binary_crossentropy'`, `metrics=['accuracy']`.
   - Class indices: `{'crack': 0, 'dent': 1}` (Sigmoid output = P(dent), P(crack) = 1 - P(dent)).
2. **Framework Isolation:** TensorFlow and PyTorch do not exchange in-memory tensor objects. All interactions occur via file paths, PIL images, or standard strings/scalars.
3. **Lazy Singleton Lifecycle:** Models are not loaded at import time to conserve RAM and prevent startup delays. Each model is loaded upon first request, cached as a thread-safe singleton.
4. **BLIP Concurrency Guard:** PyTorch BLIP generation is protected by an `asyncio.Lock` / `threading.Lock` of capacity 1 to prevent memory spikes on the 8 GB host.
5. **No External Heavy Infrastructure:** No Docker, no external database, no redis, no Node.js compilation required. Pure Python + vanilla static web assets.

---

## 2. Directory Layout and File Manifest

```
aircraft-damage-ai/
├── README.md
├── requirements.txt
├── requirements.lock.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── src/
│   └── aircraft_ai/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_setup.py
│       ├── errors.py
│       ├── data.py
│       ├── classifier.py
│       ├── train.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── blip_layer.py
│       ├── captioning.py
│       └── api.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── scripts/
│   ├── prepare_data.py
│   ├── run_train.py
│   ├── run_eval.py
│   ├── run_api.ps1
│   └── run_notebook.ps1
├── notebooks/
│   ├── original/
│   │   └── Final_Project_Classification_and_Captioning.ipynb (Read-only, SHA verified)
│   └── completed/
│       └── Final_Project_Classification_and_Captioning_completed.ipynb (Fully executed)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_data.py
│   │   ├── test_classifier.py
│   │   ├── test_blip.py
│   │   └── test_api.py
│   └── integration/
│       ├── test_end_to_end.py
│       └── test_notebook_tasks.py
├── data/ (gitignored)
│   ├── aircraft_damage_dataset_v1.tar (temp/cache)
│   └── aircraft_damage_dataset_v1/
│       ├── train/ {crack, dent}
│       ├── valid/ {crack, dent}
│       └── test/  {crack, dent}
├── artifacts/
│   ├── models/ (gitignored)
│   │   └── vgg16_aircraft_damage.keras
│   ├── results/
│   │   ├── dataset_report.json
│   │   ├── class_indices.json
│   │   ├── training_config.json
│   │   ├── history.json
│   │   ├── metrics.json
│   │   ├── blip_examples.json
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── accuracy_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── task7_prediction.png
│   │   └── test_predictions_grid.png
│   └── logs/
│       ├── data_preparation.log
│       ├── training.log
│       ├── evaluation.log
│       ├── test_run.log
│       └── notebook_execution.log
└── docs/
    ├── audit.md
    ├── plan.md
    ├── debug_log.md
    ├── report_outline.md
    └── completion_report.md
```

---

## 3. Module Breakdown

### A. Core Package (`src/aircraft_ai/`)
1. **`config.py`**:
   - Central configuration settings with environment variable overrides.
   - Seed (42), image dimensions (224, 224, 3), batch size (32), epochs (5), learning rate (0.0001).
   - Paths to data directories, model artifacts, and results.
   - P2 optional flags: data augmentation (False), fine-tuning (False), class weighting (False).
2. **`logging_setup.py`**:
   - Structured logging to both console and `artifacts/logs/`.
3. **`errors.py`**:
   - Typed domain exceptions: `DatasetNotFoundError`, `CorruptImageError`, `ModelNotTrainedError`, `InferenceError`, `BlipModelError`.
4. **`data.py`**:
   - Download logic with streaming and progress reporting.
   - Safe archive extraction (traversal rejection, size quotas).
   - Dataset validation (check 6 directories, test image readability with PIL, count samples).
   - Data leakage detection (Roboflow UUID overlap check across splits; MD5 collision detection).
5. **`classifier.py`**:
   - VGG16 base model builder with ImageNet weights, frozen backbone.
   - Custom head: `Flatten()`, `Dense(512, relu)`, `Dropout(0.3)`, `Dense(512, relu)`, `Dropout(0.3)`, `Dense(1, sigmoid)`.
   - Compiler setup: Adam(lr=1e-4), binary_crossentropy, accuracy.
6. **`train.py`**:
   - Orchestrates dataset generators (`train_datagen`, `valid_datagen`).
   - Fits model for 5 epochs.
   - Saves history, model weights (`.keras`), and config.
   - Generates loss and accuracy plots.
7. **`evaluate.py`**:
   - Computes both the notebook formula (`steps = samples // batch_size`) and the full un-truncated test evaluation.
   - Generates confusion matrix, precision, recall, F1, accuracy.
   - Produces Task 7 visualization and a deterministic test prediction grid.
8. **`inference.py`**:
   - Single image & batch inference with strict 1./255 rescaling and RGB conversion.
   - Returns probabilities for both `dent` and `crack`, class prediction, and safety disclaimer.
9. **`blip_layer.py`**:
   - Subclasses `tf.keras.layers.Layer` for `BlipCaptionSummaryLayer`.
   - Provides `generate_text(image_path, task)`.
10. **`captioning.py`**:
    - BLIP wrapper using `Salesforce/blip-image-captioning-base`.
    - Handles "caption" and "summary" prompts.
    - Thread-safe inference execution.
11. **`api.py`**:
    - FastAPI app with CORS, health check, endpoints for classification, captioning, summarization, metrics, figures, examples, and static UI file serving.

---

## 4. Acceptance Gates

| Phase | Milestone Gate | Verification Method |
| :--- | :--- | :--- |
| **Phase 1** | Workspace & Original Notebook Audited | `docs/audit.md` written, SHA-256 verified |
| **Phase 2** | Architecture and File Plan Approved | `docs/plan.md` written |
| **Phase 3** | Environment & Dataset Prepared | Dependencies installed (`pip check` passes), `scripts/prepare_data.py` runs, `artifacts/results/dataset_report.json` generated |
| **Phase 4** | VGG16 Trained, Evaluated, and Persisted | Training completes, history saved, curves plotted, test evaluation done, reload parity verified |
| **Phase 5** | BLIP Captioning & Keras Layer Verified | BLIP loads, Task 9 & Task 10 outputs generated, custom layer test passes, examples saved |
| **Phase 6** | API and Web UI Functional | FastAPI starts, endpoints respond, UI performs real inference without console errors |
| **Phase 7** | Completed Notebook & Academic Docs | Completed notebook executes with real outputs via nbclient, README and report outline completed |
| **Phase 8** | Test Suite Passes | Pytest suite passes; test log recorded in `artifacts/logs/test_run.log` |
| **Phase 9** | Debug Log Maintained | Any failures addressed and logged in `docs/debug_log.md` |
| **Phase 10**| Completion Report Verified | Every item in section R verified with explicit evidence in `docs/completion_report.md` |
