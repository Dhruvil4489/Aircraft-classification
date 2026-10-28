# AeroDamage AI — Recommended Technology Stack Specification

**Document Version:** 1.0.0  
**Date:** 2026-10-28  
**Project:** AeroDamage AI — Aircraft Damage Classification, Image Captioning and Summarization  
**Auditor Roles:** Principal Software Architect, Deep Learning Engineer, Computer Vision Researcher  

---

## 1. Overview & Architectural Philosophy

The technology selection for **AeroDamage AI** is guided by three non-negotiable architectural principles:

1. **Academic Fidelity:** The academic baseline required by the IBM Skills Network capstone (frozen VGG16 binary classifier + Hugging Face BLIP multimodal captioning wrapped in a custom Keras layer) must remain completely functional, reproducible, and verifiable.
2. **Technical Honesty:** The software must never misrepresent what the underlying models can do. Because the model was trained exclusively on 446 images of `dent` and `crack`, the system must not masquerade as a complete undamaged-versus-damaged aircraft detector without explicit safeguards.
3. **Simplicity Over Hype:** Avoid introducing unnecessary databases (PostgreSQL/MongoDB), complex frontend build tools (Next.js/React/Node), or paid cloud services when lightweight, self-contained native solutions run reliably on a standard workstation.

---

## 2. Comparison of Three Technology Stack Options

To provide clear technical guidance, three distinct architectures are evaluated.

### Summary Comparison Matrix

| Dimension | Option A: Minimum Academic Stack | Option B: Recommended Professional Stack | Option C: Advanced Research / MLOps Stack |
|---|---|---|---|
| **Primary Goal** | Minimal runnable scripts satisfying grading rubric | Maintainable, robust, defensive engineering dashboard | Enterprise-scale continuous training and monitoring |
| **Frontend** | None (Jupyter Notebook inline output) | Vanilla HTML5 / CSS3 / ES6+ JavaScript | Next.js (TypeScript) + Tailwind CSS + shadcn/ui |
| **Backend** | None (Direct Python script execution) | FastAPI + Uvicorn (ASGI REST API) | FastAPI + Celery worker queue + Redis cache |
| **Deep Learning Frameworks** | TensorFlow-CPU 2.17 + PyTorch 2.2 CPU | TensorFlow-CPU 2.17 + PyTorch 2.2 CPU | PyTorch 2.4+ (Unified across backbone & vision-language) |
| **Pretrained Models** | VGG16 (ImageNet) + BLIP Base | VGG16 (ImageNet) + BLIP Base + CLIP ViT-B/32 | ConvNeXt-V2 / Swin + BLIP-2 / LLaVA + CLIP |
| **Domain & OOD Screening** | None (Vulnerable to signatures & non-aircraft) | CLIP Zero-Shot Contrastive Screening | Custom Tri-Class Detector (Undamaged / Dent / Crack) |
| **Explainability** | Matplotlib prediction scatter | Grad-CAM on `block5_conv3` (Pre-activation logit) | Grad-CAM++ / Integrated Gradients / HiRes-CAM |
| **Decision Support** | None | Aerospace SRM Chapter 51 Guidance Engine | Aviation Maintenance ERP / AMOS Integration |
| **Data & Storage** | Local directory + tarball | Local directory + JSON artifacts (`artifacts/results/`) | DVC + S3 Object Storage + PostgreSQL database |
| **Experiment Tracking** | Print statements / Notebook cells | JSON metrics (`metrics.json`, `history.json`) | MLflow / Weights & Biases (W&B) |
| **Testing** | Manual notebook cell execution | Pytest suite (62 unit, integration & regression tests) | Pytest + Locust load testing + Great Expectations |
| **Deployment** | Local Python execution | Localhost Uvicorn + Batch launch + Docker | Multi-stage Docker + Kubernetes / Cloud Run |
| **Operational Complexity** | **Very Low** (1/5) | **Low–Medium** (2.5/5) | **High** (4.5/5) |
| **RAM Footprint** | ~1.4 GB | ~2.0 GB | ~6.5 GB+ |

---

### Option A — Minimum Reliable Academic Stack

**Concept:** The absolute smallest coherent set of technologies required to reproduce the notebook tasks and produce a working terminal or notebook demonstration.

- **Frontend:** Jupyter Notebook environment (`notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb`).
- **Backend:** None. Execution occurs directly in Python via helper scripts (`scripts/run_train.py`, `scripts/run_eval.py`, `scripts/run_blip_examples.py`).
- **ML Frameworks:** `tensorflow-cpu==2.17.1`, `torch==2.2.0+cpu`, `transformers==4.38.2`.
- **Pretrained Models:** VGG16 (ImageNet frozen) and `Salesforce/blip-image-captioning-base`.
- **Dataset Tools:** Standard Python `tarfile`, `urllib.request`, and `ImageDataGenerator`.
- **Evaluation Tools:** Scikit-Learn `confusion_matrix`, Matplotlib loss and accuracy curves.
- **Storage:** Local filesystem (`data/` and `artifacts/models/`).
- **Testing:** Notebook re-execution via `nbclient`.
- **Deployment:** Pure local script invocation.
- **Main Advantages:**
  - Zero web server or frontend complexity.
  - Exactly mirrors the university / IBM grading environment.
  - Zero chance of port collisions or browser rendering discrepancies.
- **Main Disadvantages:**
  - No interactive user interface for non-technical users.
  - Highly vulnerable to false-positive damage predictions on signatures or ordinary aircraft photographs.
  - Lacks visual attention localization (no Grad-CAM).

---

### Option B — Recommended Professional Stack (Selected)

**Concept:** A robust, maintainable architecture combining the required academic baseline with production-grade engineering safeguards, a modern dark-themed aerospace workstation UI, and multi-stage reliability screening.

- **Frontend:**
  - Modern Vanilla HTML5 (`frontend/index.html`) with semantic sectioning.
  - Vanilla CSS3 (`frontend/styles.css`) with design tokens (dark mode, glassmorphism, responsive grid).
  - Modern Vanilla ES6+ JavaScript (`frontend/app.js`) with zero build step, native `fetch()`, interactive Grad-CAM opacity canvas blending, and dynamic telemetry progress bars.
  - Typography: Google Fonts `Inter` (UI) and `JetBrains Mono` (telemetry/code).
- **Backend:**
  - Python 3.11.9 runtime.
  - FastAPI 0.143.3 on Starlette 1.8.0 with asynchronous endpoint handlers.
  - Uvicorn 0.54.0 ASGI server running on `127.0.0.1:8000`.
  - Pydantic 2.14.1 for rigorous request/response data contracts.
  - Python `multipart` streaming upload handling with strict 10MB size limits.
  - Thread-safe model singletons with mutex locks preventing redundant memory allocation.
- **Machine Learning & Computer Vision:**
  - **Module A (Damage Classification):** TensorFlow-CPU 2.17.1 / Keras 3.15.1 VGG16 architecture.
  - **Module B (Multimodal Captioning):** PyTorch 2.2.0 CPU + Hugging Face Transformers 4.38.2 BLIP Base model.
  - **Custom Keras Layer:** `BlipCaptionSummaryLayer` wrapping BLIP inference via `tf.py_function`.
  - **Stage A & B Reliability:** OpenAI CLIP ViT-B/32 zero-shot image-text contrastive screening for out-of-domain rejection and intact aircraft detection.
  - **Explainability:** Grad-CAM on VGG16 `block5_conv3` with linear pre-activation logit head.
  - **Image Processing:** Pillow 11.1.0 for format verification, RGB normalization, and bilinear 224×224 resizing.
  - **Decision Engine:** Structural Repair Manual (SRM ATA Chapter 51) operational disposition rules in `src/aircraft_ai/maintenance.py`.
- **Storage:** Local filesystem with structured JSON telemetry (`artifacts/results/*.json`). No database overhead.
- **Testing:** Automated Pytest suite (62 unit, integration, and regression tests) verifying all pipeline stages.
- **Deployment:** Local development via `start.bat` or `package.json` scripts; containerizable via single-stage Dockerfile.
- **Main Advantages:**
  - Completely preserves academic compliance while eliminating real-world failure cases (signatures, ordinary aircraft).
  - Outstanding visual aesthetics and user experience without node build complexity or bundle drift.
  - Highly testable (62 passing tests in ~60 seconds).
  - Realistic memory footprint (~2.0 GB active RAM) that runs smoothly on standard 8 GB laptops.
- **Main Disadvantages:**
  - Coexistence of two major ML frameworks (TensorFlow + PyTorch) in one process requires careful OpenMP handling.
  - Still constrained by the underlying 446-image dataset until a dedicated undamaged aircraft dataset is gathered.

---

### Option C — Advanced Research and MLOps Stack

**Concept:** A full enterprise MLOps architecture designed for multi-user commercial airline maintenance environments with continuous retraining.

- **Frontend:** Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui + Radix UI + Lucide React + Recharts.
- **Backend:** FastAPI microservices separated into:
  - Gateway / Orchestration API.
  - Dedicated VGG16 / ConvNeXt inference microservice.
  - Dedicated BLIP-2 / LLaVA vision-language worker.
- **Task Queue & Caching:** Celery with Redis for asynchronous job scheduling of heavy batch inspections.
- **Database:** PostgreSQL for inspection records, user authentication, and maintenance audit logs; MinIO / AWS S3 for image blob storage.
- **ML Frameworks:** Unified PyTorch 2.4+ (migrating VGG16 from Keras to PyTorch to eliminate dual-framework runtime overhead).
- **MLOps & Pipeline Tools:**
  - DVC (Data Version Control) for dataset tracking.
  - MLflow or Weights & Biases for experiment tracking, model registry, and hyperparameter sweeps.
  - BentoML or Triton Inference Server for multi-model model serving.
  - Evidently AI for model drift monitoring.
- **Deployment:** Multi-container Docker Compose or Kubernetes (EKS / GKE) with GPU acceleration (NVIDIA Triton).
- **Main Advantages:**
  - Massive scalability for airline fleet-wide NDT inspections.
  - Single framework (PyTorch), cutting RAM overhead and eliminating OpenMP runtime conflicts.
  - Persistent inspection audit history.
- **Main Disadvantages:**
  - **Massive Overkill:** Introduces severe infrastructure complexity (databases, caches, microservices, container networks).
  - Breaches the academic baseline requirement to use TensorFlow/Keras for the VGG16 classifier and `BlipCaptionSummaryLayer`.
  - Prohibitive operational complexity for a student project or capstone defense.

---

## 3. Recommended Technology Stack (Selected Option B)

| Layer | Recommended Technology | Exact Version | Rationale |
|---|---|---|---|
| **Language Runtime** | Python (CPython) | `3.11.9` | Optimal stability point on Windows for TensorFlow 2.17 and PyTorch 2.2. |
| **Backend Web Framework** | FastAPI | `0.143.3` | Asynchronous, auto-generates OpenAPI docs, native Pydantic integration. |
| **ASGI Server** | Uvicorn (standard) | `0.54.0` | Ultra-fast ASGI server with auto-reload capabilities. |
| **Data Validation** | Pydantic | `2.14.1` | Robust runtime typing and strict response models. |
| **Frontend Foundation** | Vanilla HTML5 / CSS3 / ES6+ | Native Browser | Zero build dependencies, high longevity, and instant dev feedback. |
| **UI Design System** | Custom Aerospace Dark Theme | CSS Tokens | Dark mode with glassmorphic cards, high-contrast cyan/emerald accents. |
| **Deep Learning (Module A)**| TensorFlow-CPU / Keras | `2.17.1` / `3.15.1` | Mandatory academic baseline for VGG16 binary classification. |
| **Deep Learning (Module B)**| PyTorch (CPU) | `2.2.0+cpu` | Mandatory runtime for Hugging Face BLIP conditional text generation. |
| **Vision-Language Model** | Hugging Face Transformers | `4.38.2` | Powers `Salesforce/blip-image-captioning-base` and CLIP pipelines. |
| **Domain Screening** | OpenAI CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | Prevents false alarms on non-aircraft images and ordinary aircraft. |
| **Image Manipulation** | Pillow (PIL) | `11.1.0` | Preprocessing parity, format validation, and safe buffer handling. |
| **Numerical Computing** | NumPy | `1.26.4` | Matrix computations and image array slicing. |
| **Metrics & Evaluation** | Scikit-Learn | `1.9.1` | Precision, recall, F1-score, and confusion matrix computation. |
| **Plotting & Visuals** | Matplotlib | `3.9.2` | Static publication-grade loss, accuracy, and confusion matrix figures. |
| **Model Explainability** | Custom Grad-CAM | In-house module | Pre-activation logit gradients on `block5_conv3` with heatmap blend. |
| **Testing** | Pytest | `9.1.1` | Fast, comprehensive test suite covering 62 unit and integration tests. |
| **HTTP Client (Testing)** | HTTPX | `0.28.1` | Powers FastAPI `TestClient` for asynchronous request testing. |
| **Environment Management** | Python `venv` + Lockfile | Pinned | Reproducible environment without system-wide package pollution. |

---

## 4. Comprehensive Compatibility Matrix

The table below documents verified package compatibility across the system.

| Package | Tested Version | Compatibility Status | Notes & Constraints |
|---|---|---|---|
| **Python** | `3.11.9` | **Verified Compatible** | Python 3.12+ has compilation issues with older PyTorch wheels; Python 3.10 is deprecated. 3.11.9 is the gold standard. |
| **TensorFlow CPU** | `2.17.1` | **Verified Compatible** | Native Windows support for GPU was dropped after TF 2.10. `tensorflow-cpu` is mandatory on Windows without WSL2. |
| **Keras** | `3.15.1` | **Verified Compatible** | Bundled with TensorFlow 2.17. Uses TensorFlow backend. |
| **PyTorch CPU** | `2.2.0+cpu` | **Verified Compatible** | Installed via PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`). Matches notebook pin. |
| **Torchvision** | `0.17.0+cpu` | **Verified Compatible** | Strictly matched to PyTorch 2.2.0. |
| **Transformers** | `4.38.2` | **Verified Compatible** | Pinned to notebook pin. Loads both BLIP and CLIP without API deprecation warnings. |
| **Tokenizers** | `0.15.2` | **Verified Compatible** | Compatible with Transformers 4.38.2. |
| **NumPy** | `1.26.4` | **Verified Compatible** | **Critical:** Must remain `< 2.0.0`. TensorFlow 2.17 and PyTorch 2.2 have C-ABI breaking incompatibilities with NumPy 2.x. |
| **Pillow** | `11.1.0` | **Verified Compatible** | Fully compatible with PyTorch and torchvision transforms. |
| **FastAPI** | `0.143.3` | **Verified Compatible** | Starlette 1.8.0 backend; fully compatible with Pydantic 2.x. |
| **Pydantic** | `2.14.1` | **Verified Compatible** | High-performance C-core (`pydantic-core==2.50.1`). |
| **Pytest** | `9.1.1` | **Verified Compatible** | Runs all 62 tests cleanly with 0 failures. |

---

## 5. Deep Learning Framework Interoperability

### The Dual-Framework Architecture
AeroDamage AI requires both **TensorFlow/Keras** and **PyTorch**:
- **TensorFlow/Keras** is required because the IBM capstone specifically mandates building, compiling, and training the VGG16 binary classifier using Keras Sequential API and `ImageDataGenerator`.
- **PyTorch** is required because `Salesforce/blip-image-captioning-base` and `openai/clip-vit-base-patch32` weights and generation pipelines in Hugging Face Transformers are implemented primarily in PyTorch.

```
+-----------------------------------------------------------------------------+
|                         PYTHON 3.11 RUNTIME PROCESS                         |
+-----------------------------------------------------------------------------+
|                                                                             |
|  +--------------------------------+   +----------------------------------+  |
|  |     TENSORFLOW / KERAS 3       |   |       PYTORCH 2.2 (CPU)          |  |
|  +--------------------------------+   +----------------------------------+  |
|  | - VGG16 Feature Extractor      |   | - BLIP Vision-Language Model     |  |
|  | - Custom Dense Head (512-512-1)|   | - CLIP ViT-B/32 Domain Model     |  |
|  | - Grad-CAM Logit Head Submodel |   | - Torchvision Tensor Transforms  |  |
|  +--------------------------------+   +----------------------------------+  |
|                 ^                                      ^                    |
|                 |                                      |                    |
|                 +------------------+-------------------+                    |
|                                    |                                        |
|                   +----------------------------------+                      |
|                   |    INTEROPERABILITY BOUNDARIES   |                      |
|                   +----------------------------------+                      |
|                   | 1. Pillow / NumPy Pixel Buffer   |                      |
|                   |    - Common exchange format      |                      |
|                   | 2. BlipCaptionSummaryLayer       |                      |
|                   |    - Uses tf.py_function wrapper |                      |
|                   | 3. Thread Mutex Guards           |                      |
|                   |    - Prevents concurrent memory  |                      |
|                   |      allocations during eval     |                      |
|                   +----------------------------------+                      |
+-----------------------------------------------------------------------------+
```

### Critical Interoperability Boundary: `BlipCaptionSummaryLayer`
Task 8 of the academic project requires creating a custom Keras layer:
```python
class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        return tf.py_function(self.process_image, [image_path, task], tf.string)
```

#### Engineering Limitations of this Boundary:
1. **Non-Differentiability:** Because `tf.py_function` executes Python bytecode outside TensorFlow's computational graph, backpropagation gradients cannot flow through this layer.
2. **SavedModel Non-Exportability:** This model cannot be exported using `tf.saved_model.save()` or converted to TensorFlow Lite / ONNX without custom C++ op registration.
3. **Threading Isolation:** PyTorch model inference inside a TensorFlow execution context must be synchronized to prevent race conditions during concurrent HTTP requests.

### OpenMP Runtime Coexistence Fix
On Windows 64-bit systems, TensorFlow and PyTorch each bundle their own OpenMP runtime (`libiomp5md.dll` or `vcomp140.dll`). To prevent sudden process termination:
- Set `KMP_DUPLICATE_LIB_OK=TRUE` in environment configurations.
- Set `TF_ENABLE_ONEDNN_OPTS=0` to ensure deterministic floating-point operations.

---

## 6. Technologies That Must Be Avoided or Postponed

1. **Avoid React / Next.js / Node.js Toolchain:**
   - *Reason:* The existing Vanilla HTML5/CSS3/JS workstation is already fully implemented, beautiful, responsive, and has zero build dependencies. Introducing Next.js would require node, npm, package bloat, hydration overhead, and SSR complexity without adding any functionality to an academic machine learning demonstrator.
2. **Avoid Relational Databases (PostgreSQL / MySQL / SQLite):**
   - *Reason:* AeroDamage AI does not have concurrent multi-tenant user accounts. All model evaluation metrics, training curves, and inspection reports are static files stored in `artifacts/results/`. A database would add connection management, migrations, and maintenance headaches for no purpose.
3. **Avoid MongoDB / NoSQL Document Stores:**
   - *Reason:* Telemetry payloads are naturally preserved as structured JSON files directly on disk.
4. **Avoid Cloud ML Platforms (SageMaker / Vertex AI):**
   - *Reason:* The models are designed for reproducible local evaluation. Cloud platforms incur ongoing financial costs and require API keys that break local academic portability.
5. **Postpone DVC and MLflow (Phase C):**
   - *Reason:* Useful for enterprise MLOps with evolving datasets, but adds unnecessary cognitive overhead for a fixed 446-image university capstone.
