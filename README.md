# AeroDamage AI: Aircraft Structural Damage Classification & Multimodal Captioning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.17.1](https://img.shields.io/badge/TensorFlow-2.17.1_CPU-orange.svg)](https://tensorflow.org/)
[![PyTorch 2.2.0](https://img.shields.io/badge/PyTorch-2.2.0_CPU-red.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset_License-CC_BY_4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end deep learning system designed to automate the visual inspection and natural language description of aircraft structural damage. The project rigorously replicates and extends the **IBM Skills Network Final Project: Classification and Captioning Aircraft Damage Using Pretrained Models** with complete academic fidelity, verified zero-mocking policy, and production-grade engineering.

---

## 1. Objectives & Problem Statement
Exterior aircraft inspection is a critical safety task routinely performed by maintenance technicians. Defects on metallic fuselages and composite panels primarily manifest as:
- **Dents:** Mechanical surface depressions resulting from ground support equipment impacts, bird strikes, or hail storms.
- **Cracks:** Linear structural fatigue discontinuities caused by cyclic pressurization or aerodynamic vibration.

Manual inspection is time-consuming and subjective. This project automates:
1. **Damage Classification (Module A):** Binary classification into `dent` vs `crack` using an ImageNet-pretrained, frozen **VGG16** feature extractor and a custom dual-dense binary classification head.
2. **Multimodal Captioning & Summarization (Module B):** Prompt-conditioned descriptive text generation using Hugging Face **BLIP** (`Salesforce/blip-image-captioning-base`) wrapped in a custom TensorFlow/Keras layer (`BlipCaptionSummaryLayer`).

---

## 2. Dataset Attribution & License
- **Original Dataset Source:** [Roboflow Universe: Aircraft Damage Detection](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk)
- **Author:** Roboflow user `youssef-donia-fhktl`
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **IBM Object Storage Archive:** `aircraft_damage_dataset_v1.tar` (SHA-256: `e8cc631b5eb6f373f6e1d47b44cdedd5fb2dd7e6ce7fc28877a1daa4a8f22bc6`)
- **Distribution Summary:**
  - **Train split:** 300 images (150 crack, 150 dent)
  - **Validation split:** 96 images (48 crack, 48 dent)
  - **Test split:** 50 images (25 crack, 25 dent)
  - **Total:** 446 images across 6 directories.

---

## 3. Architecture & Methodology

```
+-----------------------------------------------------------------------------------+
|                                 AeroDamage AI                                     |
+-----------------------------------------+-----------------------------------------+
|      Module A: VGG16 Classifier         |     Module B: BLIP Vision-Language      |
|  - Pretrained VGG16 (ImageNet, Frozen)  |  - Salesforce/blip-image-captioning-base |
|  - Flatten layer (25,088 features)      |  - PyTorch 2.2 CPU Inference            |
|  - Dense(512, ReLU) + Dropout(0.3)      |  - Caption: "This is a picture of"      |
|  - Dense(512, ReLU) + Dropout(0.3)      |  - Summary: "This is a detailed photo   |
|  - Dense(1, Sigmoid)                    |             showing"                    |
|  - Output: P(dent), P(crack)=1-P(dent)  |  - Custom Layer: BlipCaptionSummaryLayer|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         FastAPI + Vanilla Web Interface                           |
|       - Singletons & Concurrency Guards    - REST Endpoints (/api/classify, etc)  |
|       - Real Neural Net Inference Only     - Interactive Visualizations           |
+-----------------------------------------------------------------------------------+
```

### Preprocessing Parity
All images are converted to 3-channel RGB, resized to `(224, 224)` via bilinear interpolation, and rescaled identically by `1.0 / 255.0` to match the training `ImageDataGenerator(rescale=1./255)` configuration.

### Class Mapping
- Class `0`: `crack`
- Class `1`: `dent` (Sigmoid positive class)
- Threshold: `0.5`

---

## 4. Environment & Installation (Windows PowerShell)

### Prerequisites
- Windows 10/11 64-bit
- Python 3.11 installed (compatible with TensorFlow 2.17.1 CPU and PyTorch 2.2.0 CPU)

### Setup Steps
```powershell
# 1. Create Python 3.11 Virtual Environment
py -3.11 -m venv .venv

# 2. Allow script execution in current PowerShell process
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade Pip
python -m pip install --upgrade pip

# 5. Install PyTorch CPU Wheel
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# 6. Install Project Dependencies
pip install -r requirements.txt

# 7. Verify Environment Compatibility
pip check
```

---

## 5. Execution Workflow

### Step 1: Download & Verify Dataset
```powershell
python scripts/prepare_data.py
```
*Downloads the dataset safely, validates directory structure, verifies image integrity with PIL, checks for cross-split data leakage, and saves `artifacts/results/dataset_report.json`.*

### Step 2: Train VGG16 Classifier
```powershell
python scripts/run_train.py
```
*Trains for 5 epochs with batch size 32 using seed 42. Generates loss and accuracy plots and saves weights to `artifacts/models/vgg16_aircraft_damage.keras`.*

### Step 3: Run Comprehensive Evaluation
```powershell
python scripts/run_eval.py
```
*Executes both the academic truncated formula (`samples // batch_size`) and the full supplementary evaluation across all 50 test images. Generates the confusion matrix and predictions grid.*

### Step 4: Run BLIP Benchmark Captioning & Summarization
```powershell
python scripts/run_blip_examples.py
```
*Executes BLIP captioning and summarization on benchmark images and tests the custom Keras layer.*

### Step 5: Start FastAPI Service and Web Interface
```powershell
# Option A: PowerShell helper script
.\scripts\run_api.ps1

# Option B: Direct Uvicorn invocation
python -m uvicorn aircraft_ai.api:app --host 127.0.0.1 --port 8000 --reload --app-dir src
```
Open **`http://127.0.0.1:8000`** in any modern web browser to access the interactive user interface.

### Step 6: Execute Completed Jupyter Notebook End-to-End
```powershell
.\scripts\run_notebook.ps1
```
*Executes `notebooks/completed/Final_Project_Classification_and_Captioning_completed.ipynb` and writes real cell outputs directly to the file.*

---

## 6. Real Experimental Results

### Training History (5 Epochs)
- **Epoch 1:** Training Acc = 53.67%, Val Acc = 60.42%
- **Epoch 2:** Training Acc = 72.33%, Val Acc = 69.79%
- **Epoch 3:** Training Acc = 77.67%, Val Acc = 71.88%
- **Epoch 4:** Training Acc = 81.33%, Val Acc = 65.62%
- **Epoch 5:** Training Acc = **88.00%**, Val Acc = **70.83%**

### Test Set Performance
- **Academic Notebook Evaluation (Truncated 32 samples):**
  - Test Loss: `0.7326`
  - Test Accuracy: `68.75%`
- **Full Supplementary Evaluation (100% of test split / 50 samples):**
  - Test Loss: `0.5783`
  - Test Accuracy: **`76.00%`**
  - Positive Class (`dent`): Precision = `0.6970`, Recall = **`0.9200`**, F1-Score = **`0.7931`**
  - Negative Class (`crack`): Precision = `0.8824`, Recall = `0.6000`, F1-Score = **`0.7143`**
  - Confusion Matrix: `[[15, 10], [2, 23]]`

---

## 7. REST API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check and model loading state |
| `GET` | `/api/model/status` | Model training status, backbone, and runtime hardware info |
| `GET` | `/api/model/info` | Complete academic and production architectural specifications |
| `GET` | `/api/class-mapping` | Class mapping `{"crack": 0, "dent": 1}` and threshold |
| `GET` | `/api/dataset/info` | Reads `artifacts/results/dataset_report.json` |
| `POST` | `/api/analyze` | Multi-stage inspection: CLIP domain screening, VGG16 classification, Grad-CAM attention mapping, and SRM operational guidance |
| `POST` | `/api/classify` | Alias to `/api/analyze` |
| `POST` | `/api/inspection/report/html` | Generates a printable aerospace Non-Destructive Inspection (NDI) report document |
| `POST` | `/api/caption` | Multipart image upload; returns prompt-conditioned BLIP caption |
| `POST` | `/api/summary` | Multipart image upload; returns detailed BLIP summary |
| `GET` | `/api/evaluation` | Returns `history.json` and `metrics.json` |
| `GET` | `/api/examples` | Lists allow-listed test benchmark images |
| `GET` | `/api/examples/{name}` | Serves allow-listed test benchmark JPEG file |
| `GET` | `/api/results/figures/{name}` | Serves allow-listed evaluation PNG figures |

---

## 8. Test Suite Execution
Run the automated pytest test suite (39 comprehensive unit & integration tests):
```powershell
# Run entire test suite (39 tests covering inference, Grad-CAM, API, data, CLIP, BLIP)
python -m pytest -v
```

---

## 9. Important Disclaimers & Known Limitations
1. **Aviation Safety Disclaimer:** This system is an academic and educational demonstrator. It is **not certified** by the Federal Aviation Administration (FAA), European Union Aviation Safety Agency (EASA), or any regulatory authority for airworthiness determination, flight sign-off, or structural safety clearance.
2. **Captioning Semantics:** The "summary" feature is generated by conditioning the pretrained BLIP captioning model on an alternative detailed prompt (`"This is a detailed photo showing"`). It is not a separate text summarization architecture.
3. **Reproducibility:** Random seeds (`seed=42`) ensure algorithmic consistency across runs, but slight floating-point differences may occur across diverse CPU instruction architectures (e.g. AVX2/FMA).
4. **Layer Serialization:** `BlipCaptionSummaryLayer` wraps PyTorch inference inside `tf.py_function` and cannot be exported into a TensorFlow SavedModel graph.
