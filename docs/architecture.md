# AeroDamage AI — Complete System Architecture & Pipeline Specification

**Document Version:** 1.0.0  
**Date:** 2026-10-28  
**Project:** AeroDamage AI — Aircraft Damage Classification, Image Captioning and Summarization  
**Auditor Roles:** Principal Software Architect, Deep Learning Engineer, Computer Vision Researcher  

---

## 1. Architectural Philosophy & Safety Posture

AeroDamage AI automates visual non-destructive inspection (NDI) and description of aircraft structural damage. In aviation safety engineering, an automated inspection system must be **defensive by design**:

1. **Closed-World Fallacy Elimination:** A binary model trained exclusively on images of defects (`dent` and `crack`) must **never** be presented as an autonomous damage detector. Without explicit domain and damage-presence screening, it will assign high-confidence damage scores to benign aircraft surfaces or unrelated images (such as signatures).
2. **Defensive Staging:** Inspection occurs in sequential gates. If an input fails an early gate (e.g. non-aircraft or undamaged aircraft), subsequent classification heads and visual explainability heatmaps are suppressed to avoid misleading maintenance personnel.
3. **Multimodal Decoupling:** Image captioning via BLIP provides natural language surface descriptions, but is strictly decoupled from quantitative damage classification and does not serve as structural validation.

---

## 2. System Architecture Diagrams

### 2.1 Multi-Stage Aircraft Structural Damage Inspection Flow

The following sequence details how an uploaded image traverses the defensive inspection pipeline.

```mermaid
flowchart TD
    subgraph Client ["Client Tier (Browser)"]
        UI_Upload["User Uploads Inspection Photo\n(Drag & Drop or File Picker)"]
        UI_Context["Optional NDT Context\n(Aircraft Model, Panel Location,\nDefect Dimensions mm)"]
        UI_Display["Interactive Dashboard\n(Telemetry, Heatmap Slider,\nSRM Action Steps, HTML Report)"]
    end

    subgraph Gateway ["API & Validation Tier (FastAPI)"]
        API_Route["POST /api/analyze\n(Streaming Multipart Form)"]
        File_Val{"Stage 1: File Validation\n- File Size <= 10MB\n- MIME: JPEG / PNG / WebP\n- PIL Byte & Integrity Verify"}
        RGB_Norm["RGB Mode Conversion\n& Bilinear Resize (224x224)"]
    end

    subgraph StageA ["Stage A: Domain Relevance Screening"]
        CLIP_A["Zero-Shot CLIP ViT-B/32\n(Contrastive Text-Image Screening)"]
        Check_Domain{"Is Aircraft Domain?\nIn-Domain Score >= 0.50"}
        Reject_A["REJECT: UNSUPPORTED_IMAGE\nStatus: Out-of-Domain\nSuppress Classifier & Grad-CAM"]
    end

    subgraph StageB ["Stage B: Damage Presence Detection"]
        CLIP_B["Zero-Shot CLIP Structural Evidence\n(Intact Aircraft vs Close-Up Defect)"]
        Check_Presence{"Damage Evidence Present?\nDefect Score > Intact Score"}
        Flag_Intact["FLAG: DAMAGE_UNVERIFIED\nStatus: Ordinary Aircraft Photo\nSuppress Dent/Crack Head & Grad-CAM\n(Prevent False Alarms)"]
    end

    subgraph StageC ["Stage C: Defect Classification (Academic Baseline)"]
        VGG_Extract["VGG16 Backbone (Frozen)\nExtract Last Conv Feature Map (14x14x512)\nFlatten (25,088 Features)"]
        VGG_Head["Dual Dense Head\nDense(512, ReLU) + Dropout(0.3)\nDense(512, ReLU) + Dropout(0.3)\nDense(1, Sigmoid)"]
        Score_Calc["Calculate Probabilities:\nP(dent) = Sigmoid(z)\nP(crack) = 1.0 - P(dent)"]
    end

    subgraph Uncertainty ["Uncertainty & Explainability Tier"]
        Check_Margin{"Uncertainty Check:\n|P(dent) - 0.50| < 0.12 ?"}
        State_Uncertain["Status: UNCERTAIN\nBorderline Prediction\nManual NDT Review Required"]
        State_Confident["Status: COMPLETED\nClassified as DENT or CRACK"]
        GradCAM["Grad-CAM Attention Mapping\non VGG16 block5_conv3\n(Pre-activation Logit Linear Head)"]
    end

    subgraph Decision ["Maintenance Decision Support (SRM ATA 51)"]
        SRM_Engine["AeroDamage Maintenance Engine\n- Match ATA 51 Disposition Rules\n- Classify Criticality & Inspection Method\n- Identify Missing Structural Measurements\n- Generate Printable NDI Inspection Report"]
    end

    %% Flow connections
    UI_Upload --> API_Route
    UI_Context -.-> API_Route
    API_Route --> File_Val
    File_Val -- Invalid / Corrupt --> HTTP_400["HTTP 400 Bad Request"]
    File_Val -- Valid Bytes --> RGB_Norm
    RGB_Norm --> CLIP_A
    CLIP_A --> Check_Domain
    Check_Domain -- No (Signature, Face, Landscape) --> Reject_A
    Reject_A --> SRM_Engine
    Check_Domain -- Yes (Aircraft Fuselage/Panel) --> CLIP_B
    CLIP_B --> Check_Presence
    Check_Presence -- No (Ordinary Intact Aircraft) --> Flag_Intact
    Flag_Intact --> SRM_Engine
    Check_Presence -- Yes (Structural Defect Visible) --> VGG_Extract
    VGG_Extract --> VGG_Head
    VGG_Head --> Score_Calc
    Score_Calc --> Check_Margin
    Check_Margin -- Yes (Margin < 0.12) --> State_Uncertain
    Check_Margin -- No (Margin >= 0.12) --> State_Confident
    State_Uncertain --> GradCAM
    State_Confident --> GradCAM
    GradCAM --> SRM_Engine
    SRM_Engine --> UI_Display

    classDef reject fill:#ff4d4f,stroke:#d9363e,color:#fff;
    classDef warning fill:#faad14,stroke:#d48806,color:#fff;
    classDef success fill:#52c41a,stroke:#389e0d,color:#fff;
    classDef process fill:#1890ff,stroke:#096dd9,color:#fff;

    class Reject_A reject;
    class Flag_Intact,State_Uncertain warning;
    class State_Confident success;
    class VGG_Extract,VGG_Head,GradCAM,SRM_Engine process;
```

---

### 2.2 Multimodal Vision-Language Captioning & Summarization Flow (BLIP)

The image captioning and summarization pipeline runs independently of the structural classifier.

```mermaid
flowchart TD
    subgraph Input ["Image Input"]
        Img["Input Inspection Image (RGB)"]
        Task_Select{"Select Task Type\n(caption vs summary)"}
    end

    subgraph BLIP_Pipeline ["Hugging Face BLIP Pipeline"]
        Prompt_Select["Select Task Condition Prompt:\n- caption: 'This is a picture of'\n- summary: 'This is a detailed photo showing'"]
        Processor["BlipProcessor\n- Normalize & Tokenize\n- Resize to Model Resolution"]
        PyTorch_Model["BlipForConditionalGeneration\n(Salesforce/blip-image-captioning-base)\nPyTorch 2.2 CPU Inference"]
        Decode["Tokenizer Decode\n(skip_special_tokens=True)"]
    end

    subgraph Keras_Wrapper ["Academic Baseline Layer (Task 8)"]
        Keras_Layer["BlipCaptionSummaryLayer(tf.keras.layers.Layer)\nWrapped in tf.py_function\n- Non-differentiable bridge\n- Dynamic string output tensor"]
        Helper["generate_text(image_path, task)\nAcademic Helper Function"]
    end

    subgraph Output ["Client Delivery"]
        API_Endpoints["Endpoints:\nPOST /api/caption\nPOST /api/summary"]
        Result_Display["Workstation Captioning Card\nDisplay Natural Language Text"]
    end

    Img --> Task_Select
    Task_Select --> Prompt_Select
    Prompt_Select --> Processor
    Img --> Processor
    Processor --> PyTorch_Model
    PyTorch_Model --> Decode
    Decode --> Keras_Layer
    Keras_Layer --> Helper
    Helper --> API_Endpoints
    API_Endpoints --> Result_Display
```

---

## 3. Detailed Stage-by-Stage Technical Analysis

### Stage 1: File & Image Validation
- **Location:** `src/aircraft_ai/api.py:L45-74`
- **Responsibilities:**
  1. Enforces upload ceiling (`MAX_UPLOAD_SIZE_MB = 10`). Rejects payloads exceeding limit with `HTTP 413`.
  2. Verifies MIME type allowlist (`image/jpeg`, `image/png`, `image/webp`).
  3. Uses Pillow's `Image.open(io.BytesIO(file_bytes)).verify()` to confirm image header and compression integrity without executing arbitrary code.
  4. Decodes and re-encodes into strict 3-channel RGB, converting transparency channels (RGBA/LA) via white-matte alpha compositing.
- **Status:** **Implemented, Verified, and Tested.**

### Stage 2: Aircraft-Image Domain Screening (Stage A)
- **Location:** `src/aircraft_ai/domain_screening.py:L71-120`
- **Methodology:** Pretrained zero-shot contrastive image-text alignment using `openai/clip-vit-base-patch32`.
- **Target Prompt Ensembles:**
  - *In-Domain Prompts:* Close-up technical inspection photos of aircraft metal fuselage skins, wings, and composite panels.
  - *Out-of-Domain Prompts:* Signatures, handwritten text, cursive documents, human portraits, road vehicles, natural landscapes, and cartoon illustrations.
- **Decision Logic:** Softmax over pooled prompt logits. If in-domain score $< 0.50$, the image is rejected as `UNSUPPORTED_IMAGE`.
- **Status:** **Implemented, Verified, and Tested.** (Prevents the signature misclassification bug).

### Stage 3: Damage-Presence Detection (Stage B)
- **Location:** `src/aircraft_ai/domain_screening.py:L140-200`
- **The Core Problem:** The project's training dataset contains **only 446 images of structural defects (223 dent, 223 crack)**. It contains **zero images of intact, undamaged aircraft**.
- **Current Technical Implementation:**
  - Uses CLIP contrastive zero-shot screening contrasting prompts for:
    - *Intact Aircraft:* "an ordinary undamaged aircraft in flight or parked at an airport gate with clean exterior surface".
    - *Damaged Panel:* "a close-up technical inspection photograph showing structural skin damage, crack, or dent".
  - If the image depicts an intact aircraft without close-up damage features, the pipeline sets `damage_presence_status = NOT_ASSESSED`, suppresses the VGG16 binary head, withholds Grad-CAM, and outputs `STATUS_DAMAGE_UNVERIFIED`.
- **Future Production Requirement:** A genuinely certified damage-presence detector requires training a binary classifier on a balanced dataset of at least **1,000+ verified undamaged aircraft panel images** alongside damaged panels.
- **Status:** **Implemented via CLIP Screening Safeguard; Documented as Academic Limitation.**

### Stage 4: Dent-versus-Crack Classification (Stage C — Academic Baseline)
- **Location:** `src/aircraft_ai/classifier.py:L17-58`, `src/aircraft_ai/inference.py:L88-114`
- **Model Architecture:**
  - Backbone: Pretrained VGG16 on ImageNet (`include_top=False`, input shape `(224, 224, 3)`).
  - All 19 convolutional layers frozen (`trainable = False`).
  - Flatten layer: 25,088 features.
  - Dense layer 1: 512 units, ReLU activation.
  - Dropout layer 1: 30% dropout rate (`p = 0.30`).
  - Dense layer 2: 512 units, ReLU activation.
  - Dropout layer 2: 30% dropout rate (`p = 0.30`).
  - Classification head: 1 unit with Sigmoid activation ($\sigma(z) \in [0, 1]$).
- **Class Mapping:** Class `0` = `crack`, Class `1` = `dent`.
- **Decision Threshold:** $\tau = 0.50$ (predicted as `dent` if $P(\text{dent}) > 0.50$, else `crack`).
- **Status:** **Implemented, Trained, Checkpointed (`vgg16_aircraft_damage.keras`), and Verified.**

### Stage 5: Uncertainty & Borderline Decision Logic
- **Location:** `src/aircraft_ai/inference.py:L349-370`
- **Uncertainty Margin:** $\Delta = \pm 0.12$.
  $$\text{Margin} = |P(\text{dent}) - 0.50|$$
- **Decision Rules:**
  - If $\text{Margin} < 0.12$ ($P(\text{dent}) \in [0.38, 0.62]$), status is marked `UNCERTAIN`.
  - The UI highlights the result in warning amber and triggers a mandatory non-destructive testing (NDT) manual verification workflow.
- **Status:** **Implemented, Verified, and Tested.**

### Stage 6: Model Explainability via Grad-CAM
- **Location:** `src/aircraft_ai/gradcam.py:L1-216`
- **Target Layer:** Last convolutional layer of VGG16 (`block5_conv3`, feature map shape `14 × 14 × 512`).
- **Mathematical Refinement:**
  - Standard Grad-CAM on sigmoid outputs suffers from **gradient saturation** when predictions are confident ($\sigma'(z) \to 0$).
  - AeroDamage AI constructs a submodel extracting the **pre-activation logit $z$** using an auxiliary linear dense layer initialized with the trained weights:
    $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial z^c}{\partial A_{i,j}^k}$$
    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
  - Generates high-contrast jet/turbo heatmaps and blended overlays rendered directly on the frontend via HTML5 canvas with variable opacity controls.
- **Status:** **Implemented, Verified, and Tested.**

### Stage 7: BLIP Multimodal Captioning & Summarization
- **Location:** `src/aircraft_ai/captioning.py`, `src/aircraft_ai/blip_layer.py`
- **Model:** `Salesforce/blip-image-captioning-base` (~990 MB weights).
- **Prompt Conditioning:**
  - Task 9 (Caption): `"This is a picture of"`
  - Task 10 (Summary): `"This is a detailed photo showing"`
- **Custom Keras Layer:** `BlipCaptionSummaryLayer` and `generate_text(image_path, task)` strictly implemented per Task 8 of the notebook.
- **Status:** **Implemented, Verified, and Tested.**

### Stage 8: Maintenance Decision Support (SRM ATA 51)
- **Location:** `src/aircraft_ai/maintenance.py:L1-850`
- **Scope:** Translates deep learning outputs into compliant aerospace engineering guidance aligned with Boeing/Airbus Structural Repair Manual (SRM) standards:
  - Dispositions: Immediate AOG grounding, repetitive NDT inspection, or allowable operational limits.
  - Recommended NDT Methods: High-Frequency Eddy Current (HFEC) for cracks; dial depth gauges / ultrasonic thickness for dents.
  - Regulatory Disclaimers: Mandatory FAA/EASA non-certification disclosures.
- **Status:** **Implemented, Verified, and Tested.**

---

## 4. Component Boundaries & Concurrency Model

```
+--------------------------------------------------------------------------------+
|                           FASTAPI WEB SERVICE LAYER                            |
|                                                                                |
|  [POST /api/analyze]   [POST /api/caption]   [POST /api/summary]   [GET /health]
+-------------------+--------------------+--------------------+------------------+
                    |                    |                    |
                    v                    v                    v
+--------------------------------------------------------------------------------+
|                        IN-MEMORY SINGLETON & MUTEX TIER                        |
|                                                                                |
|   +-----------------------+  +-----------------------+  +-------------------+  |
|   |  VGG16 Keras Singleton|  |   CLIP Model Singleton|  |BLIP PyTorch Single|  |
|   |  (Loaded from .keras) |  |   (clip-vit-base)     |  |(blip-base)        |  |
|   |  Lock: None (Thread-  |  |   Lock: _CLIP_LOCK    |  |Lock: _BLIP_LOCK   |  |
|   |  safe predict())      |  |   (Atomic eval)       |  |(Atomic generate())|  |
|   +-----------------------+  +-----------------------+  +-------------------+  |
+--------------------------------------------------------------------------------+
                    |                    |                    |
                    v                    v                    v
+--------------------------------------------------------------------------------+
|                        PERSISTENCE & ARTIFACTS TIER                            |
|                                                                                |
|  - artifacts/models/vgg16_aircraft_damage.keras (206 MB)                       |
|  - artifacts/results/metrics.json, history.json, dataset_report.json           |
|  - data/aircraft_damage_dataset_v1/{train, valid, test} (446 Images)           |
+--------------------------------------------------------------------------------+
```

### Concurrency Guarantees
- VGG16 inference uses Keras 3 `model.predict(batch, verbose=0)` which is internally thread-safe for forward-pass execution.
- CLIP and BLIP PyTorch models use explicit Python `threading.Lock()` mutexes around forward-pass generation calls to prevent CUDA/CPU thread contention on multi-core Windows environments.
