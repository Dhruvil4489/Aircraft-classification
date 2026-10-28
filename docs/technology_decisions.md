# AeroDamage AI — Architecture Decision Records (ADRs)

**Document Version:** 1.0.0  
**Date:** 2026-10-28  
**Project:** AeroDamage AI — Aircraft Damage Classification, Image Captioning and Summarization  
**Auditor Roles:** Principal Software Architect, Deep Learning Engineer, Systems Architect  

---

## Index of Architecture Decision Records

- [ADR-01: Retention of Vanilla HTML5/CSS3/JavaScript vs React/Next.js Migration](#adr-01-retention-of-vanilla-html5css3javascript-vs-reactnextjs-migration)
- [ADR-02: Selection of FastAPI + Uvicorn as Backend Framework](#adr-02-selection-of-fastapi--uvicorn-as-backend-framework)
- [ADR-03: Dual Deep Learning Framework Coexistence (TensorFlow + PyTorch)](#adr-03-dual-deep-learning-framework-coexistence-tensorflow--pytorch)
- [ADR-04: BLIP Vision-Language Integration via Custom Keras Layer](#adr-04-blip-vision-language-integration-via-custom-keras-layer)
- [ADR-05: Zero-Shot CLIP for Domain Screening & Reliability Gates](#adr-05-zero-shot-clip-for-domain-screening--reliability-gates)
- [ADR-06: Defensive Suppression for Unverified Damage Presence](#adr-06-defensive-suppression-for-unverified-damage-presence)
- [ADR-07: Grad-CAM Implementation via Pre-Activation Logit Linear Layer](#adr-07-grad-cam-implementation-via-pre-activation-logit-linear-layer)
- [ADR-08: Local Filesystem & Structured JSON Artifacts vs Relational Database](#adr-08-local-filesystem--structured-json-artifacts-vs-relational-database)
- [ADR-09: CPU-Centric Execution on Windows vs GPU Acceleration](#adr-09-cpu-centric-execution-on-windows-vs-gpu-acceleration)
- [ADR-10: Lightweight Evaluation Artifacts vs Heavy MLOps Experiment Tracking](#adr-10-lightweight-evaluation-artifacts-vs-heavy-mlops-experiment-tracking)

---

### ADR-01: Retention of Vanilla HTML5/CSS3/JavaScript vs React/Next.js Migration

- **Status:** **Accepted**
- **Decision Type:** Mandatory
- **Selected Technology:** Native HTML5, Vanilla CSS3 (Custom Design Tokens), Vanilla Modern ES6+ JavaScript (`frontend/`).
- **Alternative Considered:** React 18, Next.js 14, Vue.js, Tailwind CSS, shadcn/ui.
- **Reason for Selection:**
  - The existing client dashboard (`index.html`, `styles.css`, `app.js`) is already fully built, styled with an aerospace engineering aesthetic, and functional.
  - Native browser code requires zero compilation, zero Node.js/npm dependencies, zero build scripts, and zero dependency vulnerability drift.
  - Adding React or Next.js to an academic ML project creates unnecessary build steps and bundle complexity without providing any user-facing benefit.
- **Compatibility Considerations:** Runs natively in any modern Evergreen web browser (Chrome, Edge, Firefox, Safari).
- **Main Trade-offs:** Manual DOM manipulation instead of declarative state management; however, the UI state is straightforward and well-structured.
- **Operational Cost / Complexity:** Zero runtime cost, zero build tooling complexity.
- **Migration Implications:** None. Retains existing verified code.

---

### ADR-02: Selection of FastAPI + Uvicorn as Backend Framework

- **Status:** **Accepted**
- **Decision Type:** Mandatory
- **Selected Technology:** FastAPI 0.143.3 on Starlette 1.8.0 with Uvicorn 0.54.0.
- **Alternative Considered:** Flask, Django, Tornado, Express.js.
- **Reason for Selection:**
  - FastAPI is the Python industry standard for machine learning inference serving.
  - Native asynchronous request handling allows the server to handle concurrent telemetry requests while CPU-bound inference runs in dedicated worker threads.
  - Built-in Pydantic integration enforces strict input validation and produces automatic OpenAPI / Swagger documentation (`/docs`).
  - Far lighter than Django; significantly faster and more modern than Flask.
- **Compatibility Considerations:** Fully compatible with Python 3.11 and Pydantic 2.x (`pydantic-core`).
- **Main Trade-offs:** None for ML inference serving.
- **Operational Cost / Complexity:** Minimal. Single process running locally on `127.0.0.1:8000`.
- **Migration Implications:** None. Existing backend is already written in FastAPI (`src/aircraft_ai/api.py`).

---

### ADR-03: Dual Deep Learning Framework Coexistence (TensorFlow + PyTorch)

- **Status:** **Accepted**
- **Decision Type:** Mandatory (Academic Requirement)
- **Selected Technology:** TensorFlow-CPU 2.17.1 (with Keras 3.15.1) AND PyTorch 2.2.0+cpu (with torchvision 0.17.0+cpu).
- **Alternative Considered:**
  1. Re-implement VGG16 in PyTorch (dropping TensorFlow).
  2. Re-implement BLIP and CLIP in TensorFlow (dropping PyTorch).
- **Reason for Selection:**
  - The university / IBM capstone rubric explicitly evaluates Keras code: `from keras.applications import VGG16`, `model.compile(...)`, `ImageDataGenerator`, and `tf.keras.layers.Layer`.
  - Migrating VGG16 to PyTorch would cause failure on academic autograders expecting Keras weights and layer names.
  - Hugging Face `Salesforce/blip-image-captioning-base` is implemented natively in PyTorch; porting it to TensorFlow is unnecessary and brittle.
  - Maintaining both frameworks satisfies the exact rubric while isolating their runtime boundaries.
- **Compatibility Considerations:**
  - Must set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows to avoid OpenMP library collisions.
  - Pinned `numpy==1.26.4` to prevent C-ABI breakage with NumPy 2.x.
- **Main Trade-offs:** Higher resident memory footprint (~2.0 GB active RAM) compared to a single-framework runtime (~1.2 GB).
- **Operational Cost / Complexity:** Requires virtual environment management and disciplined dependency pinning.
- **Migration Implications:** Preserves the required academic baseline exactly.

---

### ADR-04: BLIP Vision-Language Integration via Custom Keras Layer

- **Status:** **Accepted**
- **Decision Type:** Mandatory (Task 8 Requirement)
- **Selected Technology:** `BlipCaptionSummaryLayer(tf.keras.layers.Layer)` wrapping PyTorch inference via `tf.py_function`, invoked by helper `generate_text(image_path, task)`.
- **Alternative Considered:** Pure PyTorch inference function bypassing Keras entirely.
- **Reason for Selection:**
  - Cell 66 and Cell 68 of the academic notebook explicitly mandate implementing `BlipCaptionSummaryLayer` and `generate_text`.
  - Academic compliance requires this exact architectural artifact to be tested and demonstrated.
- **Compatibility Considerations:**
  - `tf.py_function` executes Python bytecode outside the computational graph.
  - The layer cannot be serialized to a TensorFlow SavedModel format and cannot compute gradients.
- **Main Trade-offs:** Non-exportable Keras graph; however, this layer is purely for inference and text generation, so training gradients are not required.
- **Operational Cost / Complexity:** Low. Encapsulated cleanly in `src/aircraft_ai/blip_layer.py`.
- **Migration Implications:** Fully tested and compliant with the original notebook.

---

### ADR-05: Zero-Shot CLIP for Domain Screening & Reliability Gates

- **Status:** **Accepted**
- **Decision Type:** Mandatory for Reliability (Phase 5)
- **Selected Technology:** Pretrained OpenAI CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) via Hugging Face Transformers.
- **Alternative Considered:**
  1. No domain screening (retaining vulnerability to signatures and non-aircraft photos).
  2. Training a custom out-of-domain classifier from scratch using ResNet or MobileNet.
- **Reason for Selection:**
  - Training a custom out-of-domain classifier requires thousands of negative non-aircraft images and significant manual labelling.
  - CLIP has zero-shot contrastive reasoning across millions of concepts, allowing immediate, reliable distinction between aircraft panels, signatures, handwriting, faces, and natural landscapes.
  - Operates efficiently on CPU (~80–120ms inference time).
- **Compatibility Considerations:** Runs within the existing PyTorch 2.2 / Transformers 4.38 environment.
- **Main Trade-offs:** Adds ~620 MB to active memory footprint.
- **Operational Cost / Complexity:** Moderate. Caches weights locally in Hugging Face cache.
- **Migration Implications:** Successfully resolves the reported false-positive signature bug.

---

### ADR-06: Defensive Suppression for Unverified Damage Presence

- **Status:** **Accepted**
- **Decision Type:** Mandatory for Aviation Safety & Technical Honesty
- **Selected Technology:** Pipeline gating that flags ordinary undamaged aircraft photographs as `STATUS_DAMAGE_UNVERIFIED` and suppresses the binary VGG16 classifier.
- **Alternative Considered:**
  1. Forcing the binary classifier to output a dent/crack probability on ordinary aircraft.
  2. Claiming the binary model detects damage presence.
- **Reason for Selection:**
  - The model was trained **only** on images with confirmed defects (223 dent, 223 crack). There are zero undamaged aircraft panel images in the dataset.
  - Assigning a damage score (e.g., "61.37% Crack") to an ordinary undamaged aircraft is technically false and dangerous in aerospace maintenance.
  - Defensive suppression ensures the software reports: *"Aircraft photograph accepted, but damage presence cannot be verified by this model without close-up defect evidence."*
- **Compatibility Considerations:** Fully backward compatible; returns structured JSON with clear reason codes.
- **Main Trade-offs:** The system declines to classify damage on wide-angle or ordinary aircraft photos without localized defect features.
- **Operational Cost / Complexity:** Zero additional compute; prevents critical false alarms.
- **Migration Implications:** Accurately reflects dataset realities.

---

### ADR-07: Grad-CAM Implementation via Pre-Activation Logit Linear Layer

- **Status:** **Accepted**
- **Decision Type:** Mandatory for Explainability Quality
- **Selected Technology:** Custom Grad-CAM on VGG16 layer `block5_conv3` using an auxiliary linear dense layer computing the pre-activation logit $z$.
- **Alternative Considered:**
  1. Standard Grad-CAM computing gradients of the post-sigmoid activation $\sigma(z)$.
  2. Third-party library (`tf-keras-vis` or `grad-cam`).
- **Reason for Selection:**
  - In confident binary classifications, the sigmoid function saturates, causing its derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ to vanish toward zero. This yields faint or black heatmaps.
  - Backpropagating from the pre-activation logit $z = w^T x + b$ eliminates vanishing gradients, yielding crisp, localized attention maps.
  - Implementing directly in `src/aircraft_ai/gradcam.py` avoids adding third-party dependencies.
- **Compatibility Considerations:** Uses native TensorFlow gradient tapes.
- **Main Trade-offs:** Requires splitting the model into convolutional and dense submodels in memory.
- **Operational Cost / Complexity:** Minimal computation on a 14×14 feature map.
- **Migration Implications:** Verified working with interactive opacity sliders in UI.

---

### ADR-08: Local Filesystem & Structured JSON Artifacts vs Relational Database

- **Status:** **Accepted**
- **Decision Type:** Mandatory
- **Selected Technology:** Local filesystem directory storage with JSON metadata (`artifacts/results/*.json`).
- **Alternative Considered:** SQLite, PostgreSQL, MongoDB, DynamoDB.
- **Reason for Selection:**
  - AeroDamage AI is a single-station inspection and academic demonstration project.
  - All model training logs, evaluation metrics, and class mappings are static reference artifacts.
  - Introducing PostgreSQL or SQLite adds connection pooling, schema migrations, ORM dependencies (SQLAlchemy), and process management without delivering any practical benefit.
- **Compatibility Considerations:** Cross-platform path handling via Python `pathlib.Path`.
- **Main Trade-offs:** No multi-user query indexing; entirely appropriate for current single-user workstation design.
- **Operational Cost / Complexity:** Zero database administration overhead.
- **Migration Implications:** Highly portable; entire repository can be zipped and run anywhere.

---

### ADR-09: CPU-Centric Execution on Windows vs GPU Acceleration

- **Status:** **Accepted**
- **Decision Type:** Mandatory for Local Windows Portability
- **Selected Technology:** Native Windows x86_64 CPU execution using CPU-optimized wheels (`tensorflow-cpu` and `torch+cpu`).
- **Alternative Considered:**
  1. Direct GPU acceleration via NVIDIA CUDA on Windows.
  2. WSL2 (Windows Subsystem for Linux) CUDA setup.
- **Reason for Selection:**
  - TensorFlow officially dropped native Windows GPU support after version 2.10. Running TensorFlow 2.17 with GPU on Windows requires WSL2, which adds significant virtualization setup barriers for academic evaluators.
  - The project's VGG16 model is frozen and small (14.7M parameters); single-image CPU inference takes only ~45–60 ms.
  - BLIP CPU inference takes ~400–600 ms, which is entirely acceptable for a demonstration workstation.
  - Prioritizing CPU wheels guarantees 100% out-of-the-box reproducibility on any standard Windows 10/11 laptop without CUDA driver version mismatches.
- **Compatibility Considerations:** Optimized for AVX2 instruction sets.
- **Main Trade-offs:** Batch training of 5 epochs takes ~4–6 minutes on CPU rather than ~45 seconds on GPU.
- **Operational Cost / Complexity:** Zero CUDA configuration headaches.
- **Migration Implications:** Universal compatibility across examiner machines.

---

### ADR-10: Lightweight Evaluation Artifacts vs Heavy MLOps Experiment Tracking

- **Status:** **Accepted**
- **Decision Type:** Mandatory for Academic Scope
- **Selected Technology:** Scikit-Learn evaluation scripts generating persistent JSON reports and static Matplotlib PNG figures (`artifacts/results/`).
- **Alternative Considered:** MLflow Tracking Server, Weights & Biases (W&B), Neptune.ai, DVC.
- **Reason for Selection:**
  - Cloud MLOps tools require user accounts, internet connectivity, and API keys that break offline academic grading.
  - Local MLflow server requires running a separate background tracking process and SQLite database.
  - Clean JSON files (`history.json`, `metrics.json`, `dataset_report.json`) are human-readable, git-committable, and immediately consumable by the frontend dashboard.
- **Compatibility Considerations:** Pure standard library JSON and Matplotlib.
- **Main Trade-offs:** No live training parameter comparison dashboard; however, training is run once and evaluated statically.
- **Operational Cost / Complexity:** Minimal.
- **Migration Implications:** Fully transparent and easily archived.
