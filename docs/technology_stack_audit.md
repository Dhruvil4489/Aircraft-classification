# AeroDamage AI — Complete Technology Stack Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-28  
**Project:** AeroDamage AI — Aircraft Damage Classification, Image Captioning and Summarization  
**Workspace:** `D:\Aircraft classification`  
**Auditor Roles:** Principal Software Architect, Deep Learning Engineer, Computer Vision Researcher, Backend & Frontend Architect, MLOps Engineer  

---

## 1. Executive Summary & Audit Methodology

This technology stack audit provides an exhaustive, evidence-based inventory of all software frameworks, deep learning models, libraries, system configurations, and operational workflows present in the **AeroDamage AI** workspace.

### Audit Principles
1. **Zero Assumption Policy:** Every listed technology is verified against actual files, dependency manifests, runtime environment checks (`.venv`), or git-tracked source code.
2. **Academic Baseline Preservation:** The core academic specification (frozen VGG16 binary classifier + Hugging Face BLIP captioning) from the IBM Skills Network capstone is evaluated with strict academic fidelity.
3. **Operational Honesty:** The limitations of the current dataset (only 446 images across `dent` and `crack`, with **zero undamaged aircraft images**) and the dual-framework reality (TensorFlow 2.17 + PyTorch 2.2 on Windows CPU) are documented without euphemism.

---

## 2. Complete Technology Inventory (36 Categories)

The following table catalogs all 36 architectural categories, assessing whether each technology is currently implemented, missing, optional, or unnecessary.

| # | Category | Technology | Current Status | Files & File-Level Evidence | Purpose in AeroDamage AI | Recommendation |
|---|---|---|---|---|---|---|
| **1** | **Programming Languages** | Python 3.11.9 | **Implemented & Verified** | `pyproject.toml:L10`, `.env.example:L4`, runtime check `python.exe -V` returns `3.11.9` | Primary runtime for deep learning, API service, dataset preparation, and evaluation. | **Retain**. Python 3.11.9 is the optimal bridge version supporting both TensorFlow 2.17.1 CPU and PyTorch 2.2.0 CPU on Windows. |
| | | JavaScript (ES6+ / ES2022) | **Implemented & Verified** | `frontend/app.js:L1-983` | Client-side application logic, DOM rendering, Grad-CAM canvas rendering, and REST API communication. | **Retain**. Pure modern JavaScript without transpilation avoids node toolchain complexity. |
| | | HTML5 / CSS3 | **Implemented & Verified** | `frontend/index.html:L1-933`, `frontend/styles.css:L1-950` | Semantic layout, aerospace workstation styling, CSS custom properties, and responsive layout. | **Retain**. Clean semantic markup and design tokens. |
| | | PowerShell / Batch | **Implemented & Verified** | `start.bat:L1-5`, `scripts/run_api.ps1:L1-16`, `scripts/run_notebook.ps1:L1-14`, `package.json:L6-7` | Local development orchestration, server startup, and browser launch automation. | **Retain**. Essential for Windows development workflows. |
| **2** | **Frontend Framework** | Vanilla JavaScript DOM | **Implemented & Verified** | `frontend/app.js:L7-983`, `frontend/index.html:L1-933` | Single-page application logic managing 6 workstation views, telemetry, and upload workflows. | **Retain**. Zero build step, instant reloading, and no node_modules bloat. |
| | | React / Next.js / Vue | **Unnecessary for Current Scope** | `package.json:L1-10` (no frontend dependencies installed) | Candidate frontend framework. | **Do not adopt**. For a local academic demonstration, introducing Next.js or React adds build-tool overhead without functional benefit. |
| **3** | **Frontend Language** | Standard ECMAScript 2022 | **Implemented & Verified** | `frontend/app.js` | Browser-native client logic. | **Retain**. |
| | | TypeScript | **Optional (Future)** | None (`tsconfig.json` missing) | Type safety for large client codebases. | **Postpone**. TypeScript adds unnecessary compilation overhead for a single-page student dashboard. |
| **4** | **UI & Styling Framework** | Vanilla CSS (Design System Tokens) | **Implemented & Verified** | `frontend/styles.css:L1-120` (CSS variables `--bg-primary`, `--accent-blue`, etc.) | Dark aerospace engineering theme, glassmorphism panels, responsive grid, and custom sliders. | **Retain**. Provides total control over styling with zero runtime dependencies. |
| | | Tailwind CSS | **Unnecessary for Current Scope** | None | Utility-first CSS framework. | **Do not adopt**. Vanilla CSS is already fully structured with a consistent token system. |
| **5** | **UI Component Library** | Custom Semantic Components | **Implemented & Verified** | `frontend/index.html`, `frontend/styles.css` | Aerospace status pills, telemetry cards, drop zones, confidence bars, and report tables. | **Retain**. Fully customized to aerospace NDT inspection requirements. |
| | | shadcn/ui / Radix / MUI | **Unnecessary for Current Scope** | None | Pre-built React component libraries. | **Do not adopt**. Requires React/Tailwind ecosystem. |
| **6** | **Charts & Visualization** | Matplotlib Server-Side Generation | **Implemented & Verified** | `src/aircraft_ai/evaluate.py:L150-250`, `artifacts/results/*.png` | Generates training loss curves, accuracy curves, confusion matrix, and prediction grids as static PNGs. | **Retain**. Satisfies academic project evaluation requirements directly. |
| | | HTML/CSS Dynamic Bars | **Implemented & Verified** | `frontend/app.js:L320-350`, `frontend/styles.css` | Renders dynamic confidence meters and telemetry bars in real time without external charting libraries. | **Retain**. Lightweight and performant. |
| | | Recharts / Chart.js | **Optional (Future)** | None | Interactive client-side charting. | **Optional**. Can be added later if dynamic interactive zoom on training loss curves is desired. |
| **7** | **Backend Framework** | FastAPI | **Implemented & Verified** | `requirements.txt:L14`, `src/aircraft_ai/api.py:L29-33`, `.venv` (fastapi 0.143.3) | High-performance asynchronous Python web framework serving inference APIs and static UI. | **Retain**. Industry standard for modern Python ML inference services. |
| | | Starlette | **Implemented & Verified** | Installed as core FastAPI dependency (starlette 1.8.0) | Underlying ASGI toolkit providing static file mounting and request routing. | **Retain**. |
| | | Uvicorn | **Implemented & Verified** | `requirements.txt:L15`, `package.json:L6-7`, `start.bat:L4` (uvicorn 0.54.0) | Production ASGI web server running FastAPI on `127.0.0.1:8000`. | **Retain**. |
| **8** | **API Design** | RESTful JSON & Multipart Form | **Implemented & Verified** | `src/aircraft_ai/api.py:L76-350` | Endpoints: `/health`, `/api/model/status`, `/api/analyze`, `/api/caption`, `/api/summary`, `/api/evaluation`. | **Retain**. Adheres to standard REST conventions with explicit error handling. |
| **9** | **Deep Learning Framework** | TensorFlow / Keras (CPU) | **Implemented & Verified** | `requirements.txt:L4`, `src/aircraft_ai/classifier.py:L5-80` (`tensorflow-cpu==2.17.1`, `keras==3.15.1`) | Academic baseline VGG16 model building, compilation, training, and `.keras` model persistence. | **Retain (Academic Requirement)**. Mandatory for reproducing notebook Tasks 3–7. |
| | | PyTorch (CPU) | **Implemented & Verified** | `requirements.txt:L10-11`, `src/aircraft_ai/captioning.py:L1-120` (`torch==2.2.0+cpu`, `torchvision==0.17.0+cpu`) | Powers Hugging Face BLIP multimodal captioning and CLIP zero-shot domain screening. | **Retain (Academic Requirement)**. Mandatory for BLIP inference; cannot be replaced without rewriting BLIP weights. |
| **10** | **Computer Vision Libraries** | Pillow (PIL) | **Implemented & Verified** | `requirements.txt:L5`, `src/aircraft_ai/inference.py:L47-86` (`pillow==11.1.0`) | Image file validation, decoding, RGB mode normalization, bilinear resizing, and array conversion. | **Retain**. Robust, lightweight, and universally compatible. |
| | | OpenCV (`opencv-python`) | **Missing / Optional** | Not installed in `.venv`, not in `requirements.txt` | Advanced computer vision, image morphological operations, and contours. | **Do not add unless required**. Pillow and NumPy handle all existing resizing, color normalization, and Grad-CAM blending. |
| **11** | **Pretrained Models** | VGG16 (ImageNet) | **Implemented & Verified** | `src/aircraft_ai/classifier.py:L39`, `artifacts/models/vgg16_aircraft_damage.keras` (206 MB) | Pretrained feature extractor for binary damage classification (`dent` vs `crack`). | **Retain**. Mandatory academic baseline. |
| | | Salesforce/blip-image-captioning-base | **Implemented & Verified** | `src/aircraft_ai/captioning.py:L35-36`, `src/aircraft_ai/blip_layer.py:L41-106` | Generates natural language captions (Task 9) and prompt-conditioned summaries (Task 10). | **Retain**. Mandatory academic baseline. |
| | | OpenAI CLIP (`clip-vit-base-patch32`) | **Implemented & Verified** | `src/aircraft_ai/domain_screening.py:L33-38`, `config.py:L88` | Zero-shot contrastive image-text screening for Stage A (aircraft domain) and Stage B (damage evidence). | **Retain**. Essential reliability safeguard preventing false alarms on signatures and ordinary aircraft. |
| **12** | **Image Preprocessing** | Bilinear Resize + Rescaling (1/255) | **Implemented & Verified** | `src/aircraft_ai/inference.py:L47-86`, `src/aircraft_ai/data.py:L70-120` | Preprocessing parity: converts input to 224×224×3 RGB float32 scaled to `[0.0, 1.0]`. | **Retain**. Enforces exact parity with training `ImageDataGenerator(rescale=1./255)`. |
| **13** | **Model Training** | Keras Sequential Training Pipeline | **Implemented & Verified** | `src/aircraft_ai/train.py:L1-160`, `scripts/run_train.py:L1-25` | Trains VGG16 classifier for 5 epochs with Adam (lr=1e-4) and binary cross-entropy on batch size 32. | **Retain**. Reproducible execution saving history and weights. |
| **14** | **Model Evaluation** | Scikit-Learn Metrics & Visualizations | **Implemented & Verified** | `requirements.txt:L18`, `src/aircraft_ai/evaluate.py:L1-260` (`scikit-learn==1.9.1`) | Accuracy, loss, precision, recall, F1-score, confusion matrix, and prediction grid generation. | **Retain**. Implements both academic truncated formula and full 50-image test split evaluation. |
| **15** | **Domain Screening & OOD Handling** | CLIP Zero-Shot Contrastive Screening | **Implemented & Verified** | `src/aircraft_ai/domain_screening.py:L71-226` | Rejects non-aircraft images (signatures, text, people, landscapes) and flags ordinary undamaged aircraft. | **Retain & Expand**. Core defense against out-of-distribution inputs. |
| **16** | **Model Explainability** | Grad-CAM (Pre-activation Logit Head) | **Implemented & Verified** | `src/aircraft_ai/gradcam.py:L1-216` | Computes gradients of pre-activation logit w.r.t. `block5_conv3` feature maps; outputs heatmap overlays. | **Retain**. Eliminates sigmoid gradient saturation and provides visual inspection evidence. |
| **17** | **Data Validation & Augmentation** | Safe Extraction & Split Integrity Check | **Implemented & Verified** | `scripts/prepare_data.py:L1-95`, `src/aircraft_ai/data.py:L1-280` | Verifies tar archive, checks image corruption, confirms 3 splits (train/valid/test), and checks data leakage. | **Retain**. |
| | | Runtime Data Augmentation | **Implemented (Flagged OFF by Default)** | `src/aircraft_ai/data.py:L85-115`, `.env.example:L39` | Rotation, zoom, width/height shifts, horizontal flip. | **Keep as optional flag**. Default OFF preserves exact notebook baseline; can be toggled via environment variable. |
| **18** | **Dataset Storage & Management** | Local Directory Structure | **Implemented & Verified** | `data/aircraft_damage_dataset_v1/{train,valid,test}/{crack,dent}` (446 images) | Organizes split-specific defect image files. | **Retain**. Simple and standard for `ImageDataGenerator`. |
| | | DVC (Data Version Control) | **Missing / Optional (Phase C)** | Not configured (`.dvc/` absent) | Git-integrated dataset and pipeline versioning. | **Postpone**. Unnecessary overhead for a fixed 446-image academic dataset. |
| **19** | **Model Artifact Storage** | Local Keras File Persistence | **Implemented & Verified** | `artifacts/models/vgg16_aircraft_damage.keras` (216 MB) | Single-file serialized Keras 3 model checkpoint containing architecture, weights, and optimizer state. | **Retain**. Standard Keras format. |
| | | Hugging Face Cache | **Implemented & Verified** | Local user cache `~/.cache/huggingface/hub/` | Caches weights for `blip-image-captioning-base` (~990 MB) and `clip-vit-base-patch32` (~600 MB). | **Retain**. Standard Transformers caching. |
| | | MLflow / W&B Model Registry | **Unnecessary for Current Scope** | None | Enterprise model registries. | **Do not adopt**. Unnecessary cloud infrastructure for an academic project. |
| **20** | **Database** | None (Local File Storage) | **Verified Absent (By Design)** | `artifacts/results/*.json` | System uses JSON artifacts (`metrics.json`, `dataset_report.json`, `history.json`) for data persistence. | **Do not add database**. A local academic demonstrator has no multi-user concurrent write requirements. |
| | | SQLite / PostgreSQL | **Unnecessary for Current Scope** | None | Relational database. | **Do not add**. Adding a relational database adds connection management and migrations without benefit. |
| **21** | **API Communication** | Browser `fetch()` API | **Implemented & Verified** | `frontend/app.js:L280-350` | Submits `FormData` multipart payloads and parses JSON telemetry responses. | **Retain**. Native browser API with zero bundle cost. |
| | | HTTPX | **Implemented & Verified** | `requirements.txt:L17`, `tests/` (`httpx==0.28.1`) | Async HTTP client powering FastAPI's `TestClient` for test suites. | **Retain**. |
| **22** | **File Upload Handling** | `python-multipart` & `UploadFile` | **Implemented & Verified** | `requirements.txt:L16`, `src/aircraft_ai/api.py:L45-74` (`python-multipart==0.0.32`) | Streams uploaded bytes, enforces 10MB limit, decodes via Pillow, and rejects corrupt binaries with HTTP 400. | **Retain**. |
| **23** | **Testing Frameworks** | Pytest & FastAPI TestClient | **Implemented & Verified** | `requirements-dev.txt:L2`, `pytest.ini:L1-9`, `tests/` (62 passing tests) | Executes unit, integration, and regression test suites covering inference, Grad-CAM, BLIP, CLIP, and API. | **Retain**. 62/62 verified passing tests. |
| **24** | **Code Quality & Linting** | Pyright, Black, Flake8 | **Present in Config** | `pyproject.toml:L18-25`, `requirements-dev.txt:L4-5` | Static type checking and code formatting configuration. | **Retain**. Enforces code standards. |
| **25** | **Logging & Error Monitoring** | Python Standard `logging` | **Implemented & Verified** | `src/aircraft_ai/logging_setup.py:L1-35`, `artifacts/logs/api.log` | Configures structured console and file logging with timestamps and log levels. | **Retain**. Simple, zero external dependency. |
| | | Sentry / Datadog | **Unnecessary for Current Scope** | None | Cloud application performance monitoring. | **Do not adopt**. Overkill for local execution. |
| **26** | **Experiment Tracking** | JSON Artifact Persistence | **Implemented & Verified** | `artifacts/results/history.json`, `training_config.json`, `metrics.json` | Stores training epoch metrics, evaluation results, and hyperparameter snapshots. | **Retain**. Transparent and easily inspected. |
| | | TensorBoard | **Installed but Inactive** | `tensorboard==2.17.1` installed via TensorFlow | Real-time gradient and scalar loss tracking. | **Optional**. Can be enabled during active retraining if desired. |
| | | Weights & Biases (W&B) | **Unnecessary for Current Scope** | None | Cloud experiment tracking. | **Do not adopt**. Requires external API keys and cloud account. |
| **27** | **Model Versioning** | Git Commit & Checkpoint Filenames | **Implemented & Verified** | Git repository on branch `main`, `vgg16_aircraft_damage.keras` | Tracks code revisions and model file changes. | **Retain**. |
| **28** | **Environment Management** | Python `venv` + Pinned Requirements | **Implemented & Verified** | `.venv/`, `requirements.txt`, `requirements-dev.txt`, `requirements.lock.txt` | Isolates dependencies to Python 3.11.9 on Windows. | **Retain**. Verified clean with `pip check`. |
| **29** | **Deployment & Hosting** | Localhost Execution (Uvicorn) | **Implemented & Verified** | `start.bat`, `package.json`, `src/aircraft_ai/api.py` | Runs on `http://127.0.0.1:8000`. | **Retain**. Primary target for academic defense and local evaluation. |
| **30** | **Containerization** | Docker (`Dockerfile`, `docker-compose.yml`) | **Missing (Phase 9)** | Missing in root directory | Portable multi-platform deployment container. | **Recommended to Add (Phase 9)**. A CPU-optimized Dockerfile will enable cross-platform reproducibility (Linux/macOS/Windows). |
| **31** | **CI/CD** | GitHub Actions Workflows | **Missing (Phase 9)** | Missing (`.github/workflows/` absent) | Automated test execution on commit. | **Recommended to Add (Phase 9)**. Standard lightweight workflow running `pytest`. |
| **32** | **Documentation** | Markdown Knowledge Base | **Implemented & Verified** | `README.md`, `docs/*.md` (11 existing markdown documents) | Comprehensive architectural, mathematical, and operational documentation. | **Retain & Expand**. |
| **33** | **Security** | Defensive File Validation & Path Guards | **Implemented & Verified** | `src/aircraft_ai/api.py:L45-74`, `src/aircraft_ai/config.py:L73-82` | Max 10MB upload ceiling, PIL byte verification, re-encoding to RGB, and strict file allowlists. | **Retain**. Protects against buffer overflows, file bombs, and path traversal. |
| **34** | **Performance Optimization** | Singleton Models & Concurrency Locks | **Implemented & Verified** | `src/aircraft_ai/inference.py:L20-45`, `src/aircraft_ai/domain_screening.py:L21-41`, `src/aircraft_ai/gradcam.py:L26-73` | In-memory singleton caching of heavy models (VGG16, BLIP, CLIP) with threading locks. | **Retain**. Prevents multi-gigabyte memory leaks and redundant disk I/O on every request. |
| **35** | **Accessibility & Responsive UI** | Modern CSS Grid / Flexbox & ARIA | **Implemented & Verified** | `frontend/styles.css`, `frontend/index.html` | High-contrast dark palette, responsive viewport scaling, keyboard navigation, and aria labels. | **Retain**. |
| **36** | **Git & Version Control** | Git Repository | **Implemented & Verified** | `.git/` directory, branch `main`, verified via `git status` | Source code revision tracking and experiment auditing. | **Retain**. |

---

## 3. Analysis of Existing Implementation Assets

### 3.1 Academic Notebook Assets
- **Original Notebook:** `notebooks/original/Final_Project_Classification_and_Captioning.ipynb`
  - Byte-for-byte identical to the IBM Skills Network capstone (SHA-256: `FF61DABC...`).
  - Contains 87 cells defining 10 specific student tasks.
- **Completed Notebook:** `notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb`
  - All 10 tasks solved and executed end-to-end with real outputs saved.
  - Cell 12 pinned dependencies: `pandas==2.2.3`, `tensorflow_cpu==2.17.1`, `pillow==11.1.0`, `matplotlib==3.9.2`, `transformers==4.38.2`, `torch==2.2.0+cpu`.
  - Task 8: `generate_text(image_path, task)` wrapping `BlipCaptionSummaryLayer`.
  - Task 9: Real caption output generated on test dent image `149_22_...jpg`.
  - Task 10: Real summary output generated on test dent image `149_22_...jpg`.

### 3.2 Machine Learning Checkpoints
- **VGG16 Checkpoint:** `artifacts/models/vgg16_aircraft_damage.keras` (216,264,627 bytes / 206.2 MB)
  - Trained on 300 images (150 dent, 150 crack) for 5 epochs.
  - Validation accuracy: 82.29%. Test accuracy on 50 images: 76.0%.
  - Positive class (1) = `dent`, Negative class (0) = `crack`.

### 3.3 Evaluation Visualizations & Artifacts
The following evaluation files are generated and stored in `artifacts/results/`:
- `history.json`: Epoch-by-epoch loss and accuracy metrics.
- `metrics.json`: Complete confusion matrix, precision, recall, and F1 scores.
- `dataset_report.json`: Dataset split verification, image count, and hash integrity.
- `blip_examples.json`: Verified text outputs for Task 9 and Task 10.
- `training_loss.png`, `validation_loss.png`, `accuracy_curve.png`: Academic curves.
- `confusion_matrix.png`, `test_predictions_grid.png`, `task7_prediction.png`: Evaluation plots.

---

## 4. Critical Findings & Technology Gaps

### Finding 1: The Closed-World Assumption & Dataset Limitation
- The Roboflow dataset (`aircraft_damage_dataset_v1`) contains **446 total images** divided strictly between `crack` (223) and `dent` (223).
- **There are zero images of undamaged aircraft or intact panels.**
- Consequently, the trained VGG16 model is mathematically incapable of answering: *"Is this aircraft damaged?"*
- Passing an ordinary aircraft photo to this model produces arbitrary scores (e.g., 61.37% crack, 38.63% dent).
- **Remediation implemented in workspace:** CLIP zero-shot screening (`domain_screening.py`) detects when an image is an ordinary intact aircraft and suppresses the binary classifier, setting damage presence to `NOT_ASSESSED`.

### Finding 2: Dual Framework Memory & Runtime Footprint
- The project loads both **TensorFlow 2.17.1 CPU** and **PyTorch 2.2.0 CPU** into a single Python process.
- Resident memory footprint when all models are loaded:
  - Base Python + FastAPI + Uvicorn: ~90 MB
  - VGG16 (TensorFlow): ~260 MB
  - CLIP ViT-B/32 (PyTorch): ~620 MB
  - BLIP Base (PyTorch): ~1,050 MB
  - **Total Active RAM:** ~2.0 GB
- On an 8 GB Windows machine, running all models concurrently is stable, but requires singleton caching to prevent memory exhaustion from repeated model instantiations.

### Finding 3: OpenMP Conflict Risk
- Loading both TensorFlow and PyTorch on Windows can cause runtime errors due to duplicate copies of Intel OpenMP runtime (`libiomp5md.dll`).
- **Remediation verified:** Configured `KMP_DUPLICATE_LIB_OK=TRUE` and `TF_ENABLE_ONEDNN_OPTS=0` in `.env.example` and test runners.

### Finding 4: Missing Containerization & CI/CD
- While the Python code and tests (62/62) are fully functional, the project lacks:
  1. A standard `Dockerfile` for reproducible containerized execution.
  2. A `docker-compose.yml` for single-command launching.
  3. A `.github/workflows/ci.yml` for automated testing.
- These are prioritized in Phase 9 of the implementation roadmap.
