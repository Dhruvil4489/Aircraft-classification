# AeroDamage AI — False-Positive Damage Investigation & ML Audit Report

**Date:** 2026-10-26  
**Target System:** AeroDamage AI (VGG16 Aircraft Damage Classifier)  
**Workspace:** `D:\Aircraft classification`  
**Investigation Scope:** Root-cause analysis of false-positive damage predictions on ordinary undamaged aircraft imagery.

---

## 1. Executive Summary & Problem Reproduction

### 1.1 The Reported Failure Case
An ordinary photograph of an aircraft in operational service without any visible structural crack or dent was submitted to the AeroDamage AI pipeline. The system returned:
- **Dent Score:** `38.63%` (or ~58.13% depending on framing)
- **Crack Score:** `61.37%` (or ~41.87%)
- **System Outcome:** `Result: Uncertain — manual review required` (or `CRACK IDENTIFIED` under narrower uncertainty margins)

### 1.2 Mathematical & Methodological Reproduction
1. **Model Architecture:** The classification head terminates in a single scalar neuron with a sigmoid activation function:
   $$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$
   where class 0 is mapped to `crack` and class 1 is mapped to `dent`.
2. **Probability Inversion:** In the codebase (`inference.py`):
   $$P(\text{dent}) = \sigma(z), \quad P(\text{crack}) = 1.0 - P(\text{dent})$$
3. **The Closed-World Trap:** The network was trained strictly on two classes: `crack` and `dent`. At no point in training was the network exposed to:
   - Undamaged aircraft panels or clean fuselage skins
   - Normal aircraft in hangars, aprons, or flight
   - Clean composite structures without defects
4. **Logit Behavior on Benign Imagery:**
   When an ordinary aircraft photograph (e.g., `tests/fixtures/sample_ordinary_aircraft.jpg`) is passed through the frozen VGG16 backbone and dense classification head:
   - Since the image lacks specific localized crack or dent features, the net pre-activation logit $z$ hovers near zero ($z \approx -0.463$ or $+0.328$).
   - $\sigma(-0.463) = 0.3863 \implies P(\text{dent}) = 38.63\%, P(\text{crack}) = 61.37\%$.
   - Because the previous uncertainty band was defined as $|P(\text{dent}) - 0.5| < 0.12$:
     $$|0.3863 - 0.50| = 0.1137 < 0.12$$
     The pipeline flagged the result as `UNCERTAIN — MANUAL REVIEW REQUIRED`.
5. **The Critical Flaw:** While an uncertainty flag is preferable to an unconditional false alarm, presenting scores of $61.37\%$ crack and $38.63\%$ dent to an aircraft technician strongly implies that **damage exists** and that the only open question is whether the damage is a crack or a dent.
   **A binary dent-versus-crack classifier cannot determine whether damage exists.** Presenting it as a damage detector is mathematically invalid and operationally hazardous in aerospace maintenance.

---

## 2. Comprehensive Model & Data Audit

### 2.1 Model Architecture & Checkpoint
- **Checkpoint Location:** `artifacts/models/vgg16_aircraft_damage.keras`
- **File Size:** `216,264,627 bytes` (206.2 MB)
- **Architecture:**
  - Base: VGG16 (ImageNet pretrained, `include_top=False`, input shape `(224, 224, 3)`)
  - All 19 VGG16 layers frozen (`trainable = False`)
  - Flatten layer: 25,088 features
  - Dense(512, activation='relu') + Dropout(0.3)
  - Dense(512, activation='relu') + Dropout(0.3)
  - Dense(1, activation='sigmoid')
- **Loss & Optimizer:** Binary cross-entropy with Adam optimizer ($\text{learning rate} = 10^{-4}$).

### 2.2 Dataset Audit & Class Distribution
Inspection of `data/aircraft_damage_dataset_v1`:
- **Source:** Roboflow Universe (`aircraft-damage-detection-1j9qk`, CC BY 4.0).
- **Split Distribution:**
  - `train/`: 150 crack, 150 dent (Total: 300)
  - `valid/`: 48 crack, 48 dent (Total: 96)
  - `test/`: 25 crack, 25 dent (Total: 50)
  - **Total Images:** 446 images
- **Class Balance:** Exactly 50% crack, 50% dent across all splits.
- **Undamaged / Intact Class:** **0 images**. The dataset contains **zero** verified undamaged aircraft panels or ordinary aircraft photographs.

### 2.3 Training History & Validation Performance
From `artifacts/results/history.json`:
- Epoch 1: Loss 0.7029, Accuracy 62.67% | Val Loss 0.6120, Val Accuracy 68.75%
- Epoch 2: Loss 0.4496, Accuracy 81.33% | Val Loss 0.4900, Val Accuracy 77.08%
- Epoch 3: Loss 0.4184, Accuracy 82.67% | Val Loss 0.4632, Val Accuracy 79.17%
- Epoch 4: Loss 0.4079, Accuracy 82.00% | Val Loss 0.4616, Val Accuracy 81.25%
- Epoch 5: Loss 0.3547, Accuracy 86.67% | Val Loss 0.4357, Val Accuracy 82.29%

### 2.4 Independent Test-Set Evaluation
From `artifacts/results/metrics.json` (50 samples: 25 crack, 25 dent):
- **Overall Accuracy:** `76.0%` (38/50 correct)
- **Dent Class (Positive, 1):** Precision = 69.70%, Recall = 92.00%, F1 = 79.31%
- **Crack Class (Negative, 0):** Precision = 88.24%, Recall = 60.00%, F1 = 71.43%
- **Confusion Matrix:**
  - True Crack (TN): 15
  - Crack misclassified as Dent (FP_dent): 10
  - Dent misclassified as Crack (FN_dent): 2
  - True Dent (TP): 23

### 2.5 Class Mapping and Preprocessing Parity
- **Class Indices:** `{"crack": 0, "dent": 1}`.
- **Preprocessing:** Bilinear resize to 224×224 RGB, scaled by `1.0 / 255.0`.
- Verified identical between `ImageDataGenerator(rescale=1./255)` in `train.py` and `np.asarray(img, dtype=np.float32) * (1.0 / 255.0)` in `inference.py`.

### 2.6 Model Loading & Caching Logic
- Loaded once as an in-memory singleton via `get_model()`.
- Thread-safe lazy loading with explicit model injection available for testing.

### 2.7 API & Request/Response Flow
- Fast execution through FastAPI endpoints (`/api/analyze`, `/api/classify`).
- Verified that responses return genuine model inference outputs computed dynamically on uploaded bytes; no mock data or hardcoded responses exist.

### 2.8 Grad-CAM Audit
Audit of `src/aircraft_ai/gradcam.py`:
1. **Target Layer:** Uses `block5_conv3` (last convolutional layer of VGG16, feature map shape 14×14×512). This is the correct layer for high-level semantic localization.
2. **Gradient Formulation Issue:**
   - The current code computed:
     `grads = tape.gradient(target_score, conv_out)`
     where `target_score` was the post-sigmoid activation `prob_dent` or `1.0 - prob_dent`.
   - In standard Grad-CAM theory (Selvaraju et al., 2017), gradients should be taken with respect to the **pre-activation logit** $y^c$, not the saturated sigmoid output. When $\sigma(z) \to 1.0$ or $\to 0.0$, the sigmoid gradient $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ vanishes toward 0, artificially attenuating the attention map.
3. **Misleading Presentation:**
   - Grad-CAM heatmaps were generated and displayed for **every** accepted image, including ordinary undamaged aircraft photos and uncertain results.
   - For an undamaged aircraft, Grad-CAM merely highlights whichever background feature (e.g. wing root, cockpit window, or rivet seam) contributed most to the logit. Presenting this to a user as "damage visual evidence" creates a direct visual false-positive hallucination.

---

## 3. Root Cause Classification: Confirmed vs. Hypotheses

| Hypothesis / Factor | Empirical Test & Verification | Finding | Status |
| :--- | :--- | :--- | :--- |
| **H1: Model weight corruption or stale checkpoint** | Checked SHA-256 and evaluated on test split. | Checkpoint matches 76% test accuracy and valid weights. | **Ruled Out** |
| **H2: Class mapping inversion** | Verified `class_indices.json`, `ImageDataGenerator` alphabetization, and evaluation predictions. | Class 0 = crack, Class 1 = dent verified. | **Ruled Out** |
| **H3: Preprocessing discrepancy** | Compared float32 array values between training generator and inference pipeline. | Exact parity ($[0, 1]$ RGB float32). | **Ruled Out** |
| **H4: Frontend mocks or cached results** | Inspected `frontend/app.js` and dynamic DOM injection. | Frontend genuinely displays API response; stale state cleared on upload. | **Ruled Out** |
| **RC1: Missing Negative (Undamaged) Class** | Exhaustive search of `data/aircraft_damage_dataset_v1`. | Dataset contains only 2 classes (crack & dent). Exactly 0 undamaged panels exist. | **CONFIRMED ROOT CAUSE** |
| **RC2: Problem Formulation Mismatch** | Mathematical inspection of binary sigmoid $\sigma(z) \in (0, 1)$. | Model outputs $P(\text{dent}|\text{damage})$, not $P(\text{damage}|\text{image})$. Binary classifier cannot determine damage existence. | **CONFIRMED ROOT CAUSE** |
| **RC3: Domain Screening Conflated Aircraft with Damage** | Tested CLIP prompts on ordinary aircraft vs. inspection photos. | Domain screening accepted any aircraft photo, then immediately fed it to the dent-vs-crack classifier. | **CONFIRMED ROOT CAUSE** |
| **RC4: Misleading Grad-CAM Generation on Benign Imagery** | Evaluated Grad-CAM on `sample_ordinary_aircraft.jpg`. | Generates spurious heatmaps on undamaged panels, fabricating false visual defect evidence. | **CONFIRMED ROOT CAUSE** |
| **RC5: Post-Sigmoid Grad-CAM Saturation** | Inspected `compute_gradcam` in `gradcam.py`. | Gradient taken w.r.t. post-sigmoid probability instead of pre-sigmoid logit. | **CONFIRMED ROOT CAUSE** |

---

## 4. Architectural Solution & Reformulation

To resolve the root causes without violating academic constraints:

### Stage A — Aircraft Image Validation
- Validate image integrity, decode safety, transparency alpha-compositing, resolution.
- Run contrastive zero-shot domain screening to reject non-aircraft images (signatures, documents, persons, vehicles, landscapes).
- Output: `aircraft_image_status` = `SUPPORTED` or `UNSUPPORTED`.

### Stage B — Damage Presence Detection
- Determine if the image exhibits evidence of structural surface defect or if it represents an ordinary, undamaged aircraft view / general photograph.
- Because the training dataset does not contain labelled undamaged aircraft images, a supervised model cannot be trained without fabricating data.
- For ordinary aircraft imagery where damage presence cannot be validated:
  - `damage_presence_status` = `NOT_ASSESSED`
  - Explicit operational limitation displayed:
    *"Aircraft image accepted, but damage presence cannot be reliably determined by the current dent-versus-crack classifier. No verified damage finding is available."*
  - Do NOT execute or display dent-versus-crack scores as true damage predictions.
  - Do NOT display Grad-CAM heatmaps as damage evidence.

### Stage C — Damage Type Classification
- When the image is an appropriate close-up structural inspection panel:
  - Execute the academic VGG16 baseline.
  - Output explicit fields: `aircraft_image_status`, `damage_presence_status`, `damage_type`, `dent_score`, `crack_score`, `uncertainty_status`, `model_version`.
  - Fix Grad-CAM to compute pre-sigmoid gradients and present it strictly as model explainability on classified defect panels.
