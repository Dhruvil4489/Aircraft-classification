# AeroDamage AI — Model Failure Audit & Reliability Root-Cause Report

**Audit Date:** 2026-10-20  
**Investigators:** Principal Computer Vision & AI Reliability Engineering Team  
**Target System:** AeroDamage AI (VGG16 Aircraft Damage Classifier + BLIP Multimodal Captioner)  
**Workspace:** `D:\Aircraft classification`  

---

## 1. Executive Summary: The Signature-Image Failure

### Observed Failure
When a non-aircraft image (specifically, a handwritten signature with blue/black ink strokes on a clean white background) was uploaded to the AeroDamage AI classification interface, the application returned:

```json
{
  "predicted_class": "crack",
  "probability_dent": 0.084728,
  "probability_crack": 0.915272,
  "confidence": 0.915272
}
```
In the user interface, this was displayed as:
- **Predicted Damage Type:** `CRACK`
- **Dent probability:** `8.47%` (or ~26.95% on varying stroke patterns)
- **Crack probability:** `91.53%` (or ~73.05%)
- **Result:** The system presented a completely spurious defect classification with high confidence.

### Reproduction Verification
An automated test script (`tests/fixtures/sample_signature.png`) was executed directly against `src/aircraft_ai/inference.py`:
- **Input:** 400×200 RGB image with cursive ink strokes on white background.
- **Model Loaded:** `artifacts/models/vgg16_aircraft_damage.keras`
- **Raw Sigmoid Activation:** `0.084728`
- **Inferred Output:** `crack` with 91.53% pseudo-confidence.
- **Root Cause Confirmed:** Zero domain relevance screening and zero out-of-distribution (OOD) rejection mechanism exist in the pipeline.

---

## 2. Established Root Causes (Verified by Code Inspection & Empirical Tests)

1. **Closed-World Binary Sigmoid Architecture:**
   - The classifier head terminates in `Dense(1, activation="sigmoid")`.
   - Mathematically, for any input tensor $x \in \mathbb{R}^{1 \times 224 \times 224 \times 3}$, the sigmoid function maps the logits to $(0, 1)$.
   - The code strictly partitions the universe into two states: $\text{label} = \text{"dent" if } p > 0.5 \text{ else "crack"}$.
   - There is no third class (e.g. `background`, `unknown`, `intact`, or `unsupported`).

2. **Feature Collision with High-Frequency Edge Kernels:**
   - Pretrained VGG16 convolutional layers (notably `block1_conv1` to `block3_conv3`) act as Gabor-like edge, stroke, and gradient detectors.
   - Handwritten ink strokes consist of high-contrast, linear, dark discontinuities across a bright background.
   - Aircraft surface cracks in the dataset are also characterized by thin, dark, jagged discontinuities across metallic or composite panels.
   - Consequently, the frozen VGG16 feature extractor produces high activations in crack-sensitive feature maps, pulling the dense layers towards the `crack` (class 0, sigmoid $\to 0$) output.

3. **Absence of Domain Relevance Screening:**
   - In `src/aircraft_ai/api.py`, `validate_uploaded_image()` only checks file format (JPEG/PNG/WebP) and decodability via PIL.
   - It executes no semantic checks to establish whether the image depicts aircraft exterior surfaces, fuselage panels, rivets, or metal skins.

4. **Misleading Presentation of Uncalibrated Sigmoid Scores:**
   - Raw sigmoid activations are displayed as "probabilities" (e.g., `Dent probability: 26.95%`, `Crack probability: 73.05%`).
   - In machine learning theory, a sigmoid output from an uncalibrated neural net does NOT represent a true posterior probability $P(Y=1|X)$ under covariate shift or out-of-distribution inputs.
   - Presenting uncalibrated outputs as "probabilities" violates aviation inspection reliability principles.

5. **Lack of Uncertainty & Decision Boundary Guardrails:**
   - Predictions near 0.5 (e.g. 0.48 or 0.52) are treated as definitive binary classifications without flagging uncertainty.
   - The UI supports only `dent` or `crack` banners; no states exist for `UNSUPPORTED IMAGE`, `UNCERTAIN — MANUAL REVIEW REQUIRED`, or `DOMAIN OUT-OF-DISTRIBUTION`.

---

## 3. Potential Causes Investigated and Ruled Out

| Hypothesis | Test / Verification Method | Finding | Status |
| :--- | :--- | :--- | :--- |
| **Model Checkpoint Missing or Stale** | Inspected `artifacts/models/vgg16_aircraft_damage.keras` (216 MB, SHA checked) | Real trained Keras model is present and matches the architecture. | Ruled Out |
| **Class Mapping Inverted** | Verified `class_indices.json`: `{"crack": 0, "dent": 1}` and generator alphabetization | Correctly mapped: Sigmoid represents $P(\text{dent})$, $P(\text{crack}) = 1 - P(\text{dent})$. | Ruled Out |
| **Preprocessing Discrepancy** | Compared `ImageDataGenerator(rescale=1./255)` in `train.py` vs `img * (1./255)` in `inference.py` | Both use RGB, (224, 224), bilinear resize, float32, and $1/255$ normalization. | Ruled Out |
| **Hardcoded Frontend Results** | Traced `api.py` and `app.js` fetch handlers | Results are genuinely computed by TensorFlow and passed via JSON response. | Ruled Out |
| **Model Never Trained** | Checked training curves, history, and test predictions grid | Model was trained for 5 epochs on real data; weights differ significantly from random initialization. | Ruled Out |

---

## 4. Existing Architecture & Artifact Summary

- **Backbone:** VGG16 (ImageNet pretrained, `weights="imagenet"`, `include_top=False`, all layers frozen).
- **Classification Head:**
  - `Flatten()` (Output: 25,088 features)
  - `Dense(512, activation="relu")`
  - `Dropout(0.3)`
  - `Dense(512, activation="relu")`
  - `Dropout(0.3)`
  - `Dense(1, activation="sigmoid")`
- **Trained Artifact Path:** `artifacts/models/vgg16_aircraft_damage.keras`
- **Compilation:** Adam optimizer ($\text{lr} = 10^{-4}$), Loss: `binary_crossentropy`.
- **Training Epochs:** 5 epochs, batch size 32.
- **Multimodal Captioner:** Hugging Face `Salesforce/blip-image-captioning-base` (PyTorch CPU).

---

## 5. Dataset and Evaluation Baseline

- **Dataset Source:** Roboflow Universe `aircraft-damage-detection-1j9qk` (CC BY 4.0).
- **Distribution:**
  - Train: 300 images (150 crack, 150 dent)
  - Validation: 96 images (48 crack, 48 dent)
  - Test: 50 images (25 crack, 25 dent)
  - Total: 446 images
- **Test Performance (Verified from `artifacts/results/metrics.json`):**
  - Accuracy: **76.0%** (38/50 correct)
  - Dent Precision: 74.1%, Recall: 80.0%, F1: 76.9%
  - Crack Precision: 78.3%, Recall: 72.0%, F1: 75.0%
- **Identified Dataset Weaknesses:**
  - Extreme small-sample regime (300 training samples).
  - Lack of negative/benign aircraft images (clean panels without damage).
  - High sensitivity to non-damage visual artifacts (lighting, zoom level, background hangars).

---

## 6. UI & API Integration Weaknesses

1. **Single Entry Point (`/api/classify`):** Directly triggers VGG16 prediction without multi-stage validation.
2. **Missing `/api/analyze` Workflow Endpoint:** Does not orchestrate domain screening, confidence bounding, and uncertainty evaluation.
3. **Frontend Presentation Flaws:**
   - Uses plain progress bars labeled "Crack (Class 0)" and "Dent (Class 1)" giving false certainty.
   - No workflow states for "Screening Image Domain", "Unsupported Image", or "Manual Review Required".
   - Visual aesthetic lacks the aerospace engineering precision demanded by mission-critical inspection tools.

---

## 7. Prioritized Correction Plan

### Phase A: Layered Domain Relevance & Out-of-Distribution Screening (P0)
1. **Stage 1 — Rigorous File Validation:** Verify file type, actual magic bytes, resolution bounds, transparency channels, and corrupt byte patterns.
2. **Stage 2 — Semantic Domain Screening:**
   - Implement zero-shot vision-language or dedicated domain screening using CLIP (`openai/clip-vit-base-patch32`) or vision embeddings.
   - Screen input against aircraft inspection descriptors vs. out-of-domain descriptors (handwritten text, signatures, human portraits, pets, general vehicles, screenshots, nature landscapes).
   - Establish calibrated rejection thresholds with a labelled validation suite.
3. **Stage 3 — Damage Classification:** Execute VGG16 baseline ONLY if the image passes Stage 2 domain screening.
4. **Stage 4 — Uncertainty Guard:**
   - Define uncertainty margin $\Delta = |p - 0.5|$. If $\Delta < 0.12$ (i.e. $0.38 \le p \le 0.62$), classify as `UNCERTAIN — MANUAL REVIEW REQUIRED`.

### Phase B: Benchmark Rejection Test Suite (P0)
- Add automated test cases covering:
  - Handwritten signatures (including the exact reproduced test).
  - Printed text documents & receipts.
  - Human portraits & faces.
  - Unrelated vehicles (cars, bicycles).
  - Natural landscapes & outdoor scenes.
  - Corrupt, empty, and transparent images.
  - Supported real aircraft dent and crack images.

### Phase C: Preprocessing Parity & Honest Confidence Reporting (P0)
- Enforce strict preprocessing parity checks.
- Replace misleading "probability" claims with "Model Score" and explicit uncertainty margins.
- Preserve academic baseline without modification; keep improved models in separate experiments.

### Phase D: Complete Aerospace-Grade UI Redesign (P0)
- Implement modern aerospace engineering aesthetic: deep navy/graphite palette, refined blue accents, restrained status colors, crisp typography, responsive layout.
- Build 6 dedicated pages: Overview, Damage Analysis, Captioning & Summary, Model Performance, Dataset & Quality, Model Information.
- Implement explicit analysis workflow states: Waiting $\to$ Validating $\to$ Screening Domain $\to$ Classifying $\to$ Result (Supported / Unsupported / Uncertain / Error).

---

## 8. Files to Be Modified & Rationale

| File Path | Action | Rationale |
| :--- | :--- | :--- |
| `src/aircraft_ai/domain_screening.py` | Create | Implements CLIP-based zero-shot domain relevance screening and OOD rejection. |
| `src/aircraft_ai/inference.py` | Modify | Implements multi-stage inference pipeline, uncertainty estimation, and domain-guarded classification. |
| `src/aircraft_ai/api.py` | Modify | Adds `/api/analyze` workflow endpoint, `/api/model/info`, `/api/model/status`, `/api/domain-screening/status`. |
| `src/aircraft_ai/config.py` | Modify | Defines domain thresholds, candidate domain prompt sets, uncertainty margins, and model metadata. |
| `frontend/index.html` | Redesign | Complete overhaul to aerospace workstation UI across all 6 pages. |
| `frontend/styles.css` | Redesign | Modern aerospace palette (deep navy, graphite, cyan/blue, amber, emerald, crimson), crisp typography. |
| `frontend/app.js` | Redesign | Robust client state machine, handling all workflow states (screening, unsupported, uncertain, completed). |
| `tests/unit/test_domain_screening.py` | Create | Unit tests verifying rejection of signatures, text, portraits, landscapes, and acceptance of aircraft panels. |
| `tests/integration/test_analyze_pipeline.py` | Create | End-to-end integration tests for `/api/analyze` endpoint. |
| `docs/design_research.md` | Create | Document public research on aircraft surface inspection, CLIP zero-shot screening, and UI patterns. |
| `docs/final_verification_report.md` | Create | Comprehensive verification report with evidence for all acceptance criteria. |
