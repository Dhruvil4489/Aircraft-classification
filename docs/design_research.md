# AeroDamage AI — Design & Architecture Research Report

**Research Date:** 2026-10-20  
**Authors:** AeroDamage AI Engineering & Research Team  
**Focus:** Aircraft Surface Inspection, Out-of-Distribution Defect Screening, Vision-Language Integration, and Mission-Critical Aerospace UI/UX  

---

## 1. Public References and Repositories Inspected

### 1.1 `ezedeem223/aircraft_damage_vgg16_blip`
- **Architecture:** ImageNet-pretrained VGG16 with custom binary classification head (`dent` vs `crack`) combined with Hugging Face `Salesforce/blip-image-captioning-base` for natural language report generation.
- **Interface:** Gradio-based single-page prototype.
- **Strengths:** Demonstrates basic coexistence of Keras VGG16 with PyTorch Transformers in Python.
- **Weaknesses & Critical Flaws:**
  - Complete absence of out-of-distribution (OOD) rejection. Any image (signatures, animals, documents) is forcibly classified as either dent or crack.
  - Conflates BLIP captions with defect classification ground truth.
  - Lacks uncertainty estimation, threshold calibration, and structured dataset leakage auditing.

### 1.2 `filipe-braiman/cv-aircraft-inspection`
- **Architecture:** Computer vision pipeline exploring aircraft exterior inspection, damage segmentation/classification, and preprocessing.
- **Strengths:** Focuses on image preprocessing, lighting normalization, and surface artifact mitigation.
- **Weaknesses:**
  - Closed-set assumptions: assumes all camera frames originate from pre-positioned aircraft drone/crawler rigs.
  - Lacks automated rejection for non-surface imagery or user upload errors.

### 1.3 `Afhrodite/Classification-and-Captioning-Aircraft-Damage-Using-Pretrained-Models`
- **Architecture:** Academic refactoring of the IBM Skills Network Capstone project using VGG16 and BLIP.
- **Strengths:** Strictly preserves the academic 10-task structure and dual prompt conditioning (`"This is a picture of"`, `"This is a detailed photo showing"`).
- **Weaknesses:**
  - Evaluates on truncated batches (`samples // batch_size`), omitting remaining test samples.
  - Displays uncalibrated sigmoid output as real-world probability.
  - Lacks a production-grade inspection UI and domain guardrails.

---

## 2. Technical Research: Out-of-Distribution Screening & Rejection

### 2.1 The Open-Set Dilemma in Defect Inspection
In aerospace maintenance, an automated inspection tool deployed in the field will encounter arbitrary inputs:
- Inspection logs, handwritten maintenance signatures, job sheets, calibration barcodes.
- Non-aircraft photographs (hangar floors, tools, personnel portraits, vehicles, outdoor runways).
- Out-of-focus, corrupt, or unreadable sensor captures.

A standard binary classifier ($f: \mathcal{X} \to \{0, 1\}$) maps all inputs into its two known classes. When presented with a handwritten signature, high-frequency stroke features inadvertently activate crack-sensitive convolutional filters in VGG16, resulting in a spurious high-confidence `crack` prediction.

### 2.2 Contrastive Vision-Language Domain Screening (CLIP)
To solve this without needing tens of thousands of negative training samples, modern industrial systems employ **Contrastive Language-Image Pretraining (CLIP)** zero-shot screening:
1. **Multi-Modal Alignment:** CLIP embeds image features and text features into a common latent hypersphere $\mathcal{S}^{d-1}$.
2. **Dual-Set Prompt Ensembling:**
   - **In-Domain Anchor Prompts ($\mathcal{T}_{\text{in}}$):**
     - `"a close-up photograph of aircraft exterior surface with structural damage"`
     - `"an aircraft metal fuselage panel showing surface crack or dent"`
     - `"aircraft metallic skin with rivet lines and structural defect"`
     - `"a technical photo of airplane composite panel defect inspection"`
   - **Out-of-Domain Reject Prompts ($\mathcal{T}_{\text{out}}$):**
     - `"a handwritten signature or cursive ink writing on paper"`
     - `"a printed text document, receipt, or paperwork"`
     - `"a photograph of a human face or portrait"`
     - `"a motor vehicle, car, truck, or bicycle"`
     - `"an animal, cat, or dog"`
     - `"a natural landscape, forest, mountains, or scenic sky"`
     - `"a digital drawing, screenshot, diagram, or graphic artwork"`
3. **Domain Relevance Metric ($S_{\text{domain}}$):**
   $$S_{\text{domain}}(x) = \frac{\sum_{t \in \mathcal{T}_{\text{in}}} \exp(\text{sim}(I(x), T(t)) / \tau)}{\sum_{t \in \mathcal{T}_{\text{in}} \cup \mathcal{T}_{\text{out}}} \exp(\text{sim}(I(x), T(t)) / \tau)}$$
   If $S_{\text{domain}} < \theta_{\text{rejection}}$, the pipeline triggers **`UNSUPPORTED IMAGE — DOMAIN OUT-OF-DISTRIBUTION`** and terminates before invoking the VGG16 classifier.

### 2.3 Uncertainty Decision Boundary
Even for images that pass domain screening, a classification score $p \in [0.38, 0.62]$ represents high epistemic and aleatoric uncertainty. The system must not force a binary choice. It triggers **`UNCERTAIN — MANUAL REVIEW REQUIRED`**.

---

## 3. UI/UX Design Patterns for Aerospace Inspection Workstations

### 3.1 Design Aesthetic: Aerospace Engineering Workstation
Unlike generic web apps or colorful consumer AI dashboards, mission-critical aerospace inspection workstations (e.g. Airbus NDT systems, Boeing flight-line diagnostic tools) adhere to strict information hierarchy:
- **Palette:**
  - Base: Deep Navy (`#0A1128`, `#101F42`) and Slate Graphite (`#1A2536`, `#232F42`)
  - Accent / Focus: Aerospace Precision Blue (`#2563EB`, `#38BDF8`)
  - Validation / Normal: Restrained Emerald (`#059669`, `#10B981`)
  - Uncertainty / Warning: Aviation Amber (`#D97706`, `#F59E0B`)
  - Error / Defect Discontinuity: Controlled Crimson (`#DC2626`, `#EF4444`)
- **Typography:** Inter (for clean, legible UI hierarchies) and JetBrains Mono (for telemetry, probabilities, dimensions, and model metadata).
- **Layout:** High-density, structured split-view inspection viewport with dedicated diagnostic panels.

### 3.2 Six Core Functional Views
1. **Overview:** Executive operational status, verified model weights path, dataset scale, real test accuracy, and educational safety disclaimers.
2. **Damage Analysis:** Primary inspection workspace with drag-and-drop, pre-verified test samples, 4-stage pipeline execution indicators, domain screening telemetry, and honest score display.
3. **Captioning & Summary:** Dedicated multimodal workbench evaluating BLIP conditional generation with separate prompt controls.
4. **Model Performance:** Real confusion matrix, ROC-AUC, Precision-Recall, per-class F1-scores, and training loss/accuracy curves.
5. **Dataset & Quality:** Real split counts (300/96/50), class distributions, data leakage hash audit, Roboflow attribution, and CC BY 4.0 license.
6. **Model Information:** Full architectural specification, layer dimensions, frozen backbone details, preprocessing parity proofs, and known limitations.

---

## 4. Architectural Decisions Adopted for AeroDamage AI

| Dimension | Previous State | New Architecture |
| :--- | :--- | :--- |
| **Domain Screening** | None (forces all images to Dent/Crack) | Layered CLIP zero-shot domain relevance filter with calibrated threshold $\theta = 0.50$. |
| **Output Taxonomy** | Binary: `dent` or `crack` | 5 States: `DENT`, `CRACK`, `UNSUPPORTED IMAGE`, `UNCERTAIN — MANUAL REVIEW`, `ANALYSIS FAILED`. |
| **Score Semantics** | Displayed as "Real Probabilities" | Displayed as "Raw Model Score" with confidence margins and explicit uncertainty callouts. |
| **API Endpoints** | Direct `/api/classify` | Multi-stage `/api/analyze` coordinating file validation, domain screening, classification, and metadata. |
| **Dashboard UI** | Basic student CRUD layout | High-performance aerospace inspection workstation with responsive CSS tokens. |
| **Evaluation Reporting** | Truncated notebook metric only | Dual evaluation: Academic truncated formula + full 50-image non-truncated test evaluation. |
