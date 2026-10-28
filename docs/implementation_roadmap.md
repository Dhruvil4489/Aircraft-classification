# AeroDamage AI — Phased Implementation & Engineering Roadmap

**Document Version:** 1.0.0  
**Date:** 2026-10-28  
**Project:** AeroDamage AI — Aircraft Damage Classification, Image Captioning and Summarization  
**Auditor Roles:** Principal Software Architect, Deep Learning Engineer, MLOps Engineer  

---

## 1. Roadmap Architecture & Execution Principles

This roadmap outlines the systematic sequence of engineering phases required to maintain, verify, stabilize, and deploy **AeroDamage AI**.

### Core Execution Rules
1. **Academic Baseline First:** Phases 1–4 establish and verify absolute parity with the IBM Skills Network capstone specification.
2. **Defensive Reliability Extension:** Phases 5–6 address real-world vulnerabilities (signature false positives, ordinary aircraft handling, explainability).
3. **Engineering Polish & Verification:** Phases 7–9 ensure client responsiveness, full automated test coverage, and reproducible deployment.
4. **No Premature Framework Migration:** A major framework migration (e.g. moving entirely to PyTorch or introducing React) is explicitly discouraged unless the project scope fundamentally expands beyond the academic capstone.

---

## 2. Phased Implementation Roadmap (Phases 1 – 9)

```
Phase 1: Stack & Asset Verification
   |
   v
Phase 2: Dependency & Runtime Coexistence
   |
   v
Phase 3: VGG16 Classification Pipeline
   |
   v
Phase 4: BLIP Multimodal Captioning & Keras Layer
   |
   v
Phase 5: Image Validation, Domain & Presence Screening
   |
   v
Phase 6: Evaluation, Calibration & Explainability
   |
   v
Phase 7: Frontend/Backend Workstation Integration
   |
   v
Phase 8: Automated Regression Testing (Pytest)
   |
   v
Phase 9: Documentation, Containerization & Handover
```

---

### Phase 1: Verify Current Stack & Existing Application

- **Objective:** Establish ground-truth baseline by inspecting all repository files, confirming environment integrity, verifying the original notebook checksum, and auditing existing assets.
- **Relevant Technologies:** Python 3.11.9, PowerShell, Git, SHA-256 hashing tools.
- **Prerequisites:** Workspace cloned with read permissions on Windows 11.
- **Expected Deliverables:**
  - `docs/technology_stack_audit.md` (Completed).
  - SHA-256 verification of `notebooks/original/Final_Project_Classification_and_Captioning.ipynb`.
  - Virtual environment check confirming `.venv` with Python 3.11.9.
- **Identified Risks:**
  - Modifying or corrupting the original notebook during inspection.
- **Verification Criteria:**
  - Original notebook SHA-256 matches `FF61DABC2DC3477C8A69B85815F9D25A695515D31C0B0C774603CC59B86FD7AC`.
  - `git status` confirms zero untracked data corruption.

---

### Phase 2: Resolve Dependency and Model Compatibility Issues

- **Objective:** Stabilize package manifests and resolve runtime coexistence conflicts between TensorFlow 2.17 CPU and PyTorch 2.2 CPU on Windows.
- **Relevant Technologies:** `pip`, `requirements.txt`, `requirements-dev.txt`, `requirements.lock.txt`, OpenMP runtime flags.
- **Prerequisites:** Phase 1 complete.
- **Expected Deliverables:**
  - Pinned `numpy==1.26.4` (blocking NumPy 2.x breaking changes).
  - Pinned `tensorflow-cpu==2.17.1` and `torch==2.2.0+cpu`.
  - Validated `.env.example` with `KMP_DUPLICATE_LIB_OK=TRUE` and `TF_ENABLE_ONEDNN_OPTS=0`.
- **Identified Risks:**
  - Windows OpenMP collision crashing Python interpreter on simultaneous TF/Torch import.
  - NumPy 2.0 wheel auto-upgrade breaking TensorFlow C-extensions.
- **Verification Criteria:**
  - `python -m pip check` outputs `No broken requirements found.`
  - Python test script successfully imports both `tensorflow` and `torch` in the same process without errors.

---

### Phase 3: Stabilize VGG16 Training and Inference Pipeline

- **Objective:** Validate dataset extraction, class mapping (`{"crack": 0, "dent": 1}`), preprocessing parity (224×224 bilinear, rescale 1/255), VGG16 architecture, and training reproducibility.
- **Relevant Technologies:** TensorFlow/Keras 3, Pillow, Scikit-Learn.
- **Prerequisites:** Phase 2 complete.
- **Expected Deliverables:**
  - Dataset preparation script: `scripts/prepare_data.py`.
  - Training script: `scripts/run_train.py` (5 epochs, batch size 32, seed 42, Adam lr=1e-4).
  - Model checkpoint: `artifacts/models/vgg16_aircraft_damage.keras` (206 MB).
  - Baseline inference module: `src/aircraft_ai/inference.py`.
- **Identified Risks:**
  - Inverted class indices causing dent predictions to be labelled cracks.
  - Preprocessing mismatch between training `ImageDataGenerator` and inference NumPy scaling.
- **Verification Criteria:**
  - Model trains for 5 epochs and achieves validation accuracy $> 70\%$.
  - Unit tests in `tests/unit/test_classifier.py` and `test_data.py` pass cleanly.

---

### Phase 4: Integrate BLIP Captioning and Summary Generation

- **Objective:** Implement Hugging Face BLIP multimodal captioning, condition prompts for captioning vs summarization, and build the custom Keras layer mandated by Task 8.
- **Relevant Technologies:** PyTorch 2.2 CPU, Hugging Face Transformers 4.38.2, `Salesforce/blip-image-captioning-base`.
- **Prerequisites:** Phase 3 complete.
- **Expected Deliverables:**
  - Captioning module: `src/aircraft_ai/captioning.py`.
  - Custom Keras layer: `src/aircraft_ai/blip_layer.py` (`BlipCaptionSummaryLayer` and `generate_text`).
  - Batch demonstration script: `scripts/run_blip_examples.py`.
  - Artifact record: `artifacts/results/blip_examples.json`.
- **Identified Risks:**
  - `tf.py_function` type incompatibility with Keras 3 dynamic tensor execution.
  - Excessive download latency (~990 MB weights) during first load.
- **Verification Criteria:**
  - `generate_text(image_path, "caption")` generates natural caption starting with `"This is a picture of"`.
  - `generate_text(image_path, "summary")` generates detailed description starting with `"This is a detailed photo showing"`.
  - Unit tests in `tests/unit/test_blip.py` pass cleanly.

---

### Phase 5: Implement Defensive Validation, Domain Screening & Damage Presence

- **Objective:** Eliminate catastrophic false alarms on out-of-domain images (signatures, text, portraits) and ordinary undamaged aircraft photographs.
- **Relevant Technologies:** OpenAI CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), Pillow safe decoding.
- **Prerequisites:** Phase 4 complete.
- **Expected Deliverables:**
  - Defensive upload validation in `src/aircraft_ai/api.py`.
  - CLIP zero-shot domain screening in `src/aircraft_ai/domain_screening.py`.
  - Damage-presence gating logic in `src/aircraft_ai/inference.py:analyze_image()`.
  - ATA Chapter 51 maintenance recommendations in `src/aircraft_ai/maintenance.py`.
- **Identified Risks:**
  - Over-filtering valid but poorly lit aircraft damage images.
  - Inappropriate confidence claims on non-damage images.
- **Verification Criteria:**
  - Submitting `tests/fixtures/sample_signature.png` yields `status = UNSUPPORTED_IMAGE` with zero damage classification.
  - Submitting `tests/fixtures/sample_ordinary_aircraft.jpg` yields `status = DAMAGE_UNVERIFIED` and suppresses the binary head.

---

### Phase 6: Improve Model Evaluation and Uncertainty Reporting

- **Objective:** Implement rigorous statistical evaluation, dual-mode evaluation (notebook truncated steps vs full 50-image test set), calibrated uncertainty margins, and Grad-CAM explainability.
- **Relevant Technologies:** Scikit-Learn 1.9.1, Matplotlib 3.9.2, TensorFlow gradient tapes.
- **Prerequisites:** Phase 5 complete.
- **Expected Deliverables:**
  - Evaluation script: `scripts/run_eval.py` producing `metrics.json`, `confusion_matrix.png`, and `test_predictions_grid.png`.
  - Grad-CAM module: `src/aircraft_ai/gradcam.py` with pre-activation logit linear head.
  - Calibrated decision margin logic ($|P(\text{dent}) - 0.5| < 0.12 \implies \text{UNCERTAIN}$).
- **Identified Risks:**
  - Sigmoid gradient vanishing saturating Grad-CAM heatmaps (resolved by logit submodel).
- **Verification Criteria:**
  - Full test split evaluation matches verified performance: 76.0% accuracy (38/50 correct).
  - Grad-CAM generates valid base64 PNG data URLs highlighting defect regions.

---

### Phase 7: Complete Frontend/Backend Integration & Responsive UI

- **Objective:** Finalize the interactive aerospace engineering workstation UI, connecting all 6 dashboard sections to live FastAPI REST endpoints.
- **Relevant Technologies:** HTML5, CSS3 (Design Tokens), Vanilla JavaScript (ES6+), FastAPI StaticFiles.
- **Prerequisites:** Phase 6 complete.
- **Expected Deliverables:**
  - Workstation views: Overview, Damage Analysis, Image Captioning, Model Performance, Dataset & Quality, Model Information.
  - Live Grad-CAM overlay slider with dynamic canvas alpha blending.
  - Printable NDI inspection report generator (`/api/inspection/report/html`).
  - Startup automation: `start.bat` and `scripts/run_api.ps1`.
- **Identified Risks:**
  - Cross-Origin Resource Sharing (CORS) blocks during development.
  - UI layout breakage on smaller tablet/laptop screens.
- **Verification Criteria:**
  - Navigating to `http://127.0.0.1:8000` renders workstation without console errors.
  - File upload, analysis, heatmap toggle, and report export execute seamlessly.

---

### Phase 8: Add Automated Tests & Continuous Verification

- **Objective:** Establish an automated, regression-proof test suite covering unit, integration, and end-to-end user journeys.
- **Relevant Technologies:** Pytest 9.1.1, FastAPI TestClient, HTTPX.
- **Prerequisites:** Phase 7 complete.
- **Expected Deliverables:**
  - Unit test suite: `tests/unit/` (9 test modules).
  - Integration test suite: `tests/integration/` (2 test modules).
  - End-to-end notebook task validation: `tests/integration/test_notebook_tasks.py`.
  - Test fixtures: `tests/fixtures/sample_signature.png`, `sample_ordinary_aircraft.jpg`.
- **Identified Risks:**
  - Slow test execution due to heavy model loading (mitigated by singleton fixtures).
- **Verification Criteria:**
  - Running `python -m pytest -v` runs all **62 tests with 100% pass rate** in under 75 seconds.

---

### Phase 9: Finalize Documentation, Containerization & Handover

- **Objective:** Package the project for seamless evaluation, grading, and external deployment, including containerization assets.
- **Relevant Technologies:** Docker, Dockerfile, Docker Compose, Markdown documentation.
- **Prerequisites:** Phase 8 complete.
- **Expected Deliverables:**
  - Project documentation in `docs/`:
    - `technology_stack_audit.md`
    - `recommended_technology_stack.md`
    - `architecture.md`
    - `technology_decisions.md`
    - `implementation_roadmap.md`
  - Single-stage CPU-optimized `Dockerfile` and `docker-compose.yml`.
  - GitHub Actions test automation workflow (`.github/workflows/ci.yml`).
- **Identified Risks:**
  - Multi-gigabyte Docker image size due to PyTorch and TensorFlow wheels.
- **Verification Criteria:**
  - Clean container build and execution.
  - Complete, verifiable documentation answering all academic evaluation criteria.
