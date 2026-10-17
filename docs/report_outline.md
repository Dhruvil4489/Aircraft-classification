# Academic Report Outline: Aircraft Damage Classification and Multimodal Captioning

**Title:** Automated Aircraft Damage Detection, Classification, and Vision-Language Description Using Deep Transfer Learning (VGG16) and Transformer-Based Image Captioning (BLIP)  
**Authors:** IBM Skills Network Lab Framework (Adapted and Executed by Antigravity AI)  
**Workspace:** `D:\Aircraft classification`  
**Date:** 2026-10-17  

---

## 1. Abstract
Automated visual inspection of aircraft exterior structures provides an essential mechanism to augment manual maintenance operations, reducing inspection turnaround times and mitigating human subjectivity. This report presents an end-to-end deep learning framework combining computer vision classification and multimodal vision-language generation. Module A utilizes an ImageNet-pretrained, frozen **VGG16** convolutional backbone coupled with a dual-dense binary classification head (512-512-1 units) to categorize surface defects into **dent** or **crack** damage classes. Module B implements **Hugging Face BLIP** (`Salesforce/blip-image-captioning-base`) wrapped in a custom TensorFlow/Keras layer (`BlipCaptionSummaryLayer`) to produce natural language descriptions and prompt-conditioned summaries. On an independently evaluated test split of 50 images, the VGG16 model achieved **76.00% overall accuracy**, with a positive class (dent) **F1-score of 0.7931** (precision: 0.6970, recall: 0.9200) and crack F1-score of 0.7143. Truncated academic evaluation per the original course formula (`samples // batch_size = 1 step`) yielded a baseline test accuracy of **68.75%** on the first 32 samples. Data leakage analysis confirmed zero cross-split exact duplicates and zero Roboflow base image identifier overlaps.

---

## 2. Introduction & Problem Statement
Commercial aviation safety standards require rigorous pre-flight and maintenance inspections of aircraft skins, engine cowlings, and control surfaces. Manual visual inspections are labor-intensive, subject to inspector fatigue, and qualitative. Surface defects typically present in two predominant forms:
1. **Cracks:** Linear material discontinuities caused by structural fatigue, cyclic pressurization stress, or vibration. If undetected, cracks propagate and cause catastrophic airframe failure.
2. **Dents:** Localized surface deformations caused by ground handling impacts, bird strikes, or hail storms. Dents compromise aerodynamic laminar flow and may conceal sub-surface composite delamination.

Automating this triage using lightweight computer vision models provides rapid diagnostic support for ground maintenance technicians.

---

## 3. Project Objectives
1. Implement and verify a robust, reproducible deep transfer learning classification pipeline utilizing VGG16.
2. Maintain strict fidelity to the original academic specifications across all 10 notebook tasks.
3. Quantify classification performance using both the notebook's truncated batch evaluation and a non-truncated full-dataset evaluation with confusion matrix analysis.
4. Integrate multimodal image captioning and summarization using Hugging Face BLIP within a custom Keras layer.
5. Provide a production-grade FastAPI microservice and interactive web UI for real-time inference without synthetic demo data.

---

## 4. Dataset Description & Integrity Audit
- **Source:** Roboflow Universe ("Aircraft Damage Detection" by user `youssef-donia-fhktl`, License: CC BY 4.0). Hosted via IBM Cloud Object Storage.
- **Archive:** `aircraft_damage_dataset_v1.tar` (SHA-256: `e8cc631b5eb6f373f6e1d47b44cdedd5fb2dd7e6ce7fc28877a1daa4a8f22bc6`).
- **Class Mapping:** `{"crack": 0, "dent": 1}`.
- **Sample Distribution:**
  - **Train:** 300 total (150 crack, 150 dent) — 50.0% / 50.0% balanced.
  - **Validation:** 96 total (48 crack, 48 dent) — 50.0% / 50.0% balanced.
  - **Test:** 50 total (25 crack, 25 dent) — 50.0% / 50.0% balanced.
  - **Total Dataset Size:** 446 images.
- **Integrity & Leakage Findings:**
  - PIL Image Verification: 446 / 446 images valid and readable. Zero corrupted or unsupported files.
  - Cross-Split Base Identifier Overlaps: 0.
  - Cross-Split Exact Hash Duplicates: 0.

---

## 5. Methodology & Architecture

### 5.1 Module A: VGG16 Feature Extractor & Classification Head
- **Backbone:** VGG16 initialized with weights pretrained on ImageNet (`include_top=False`, input shape `224x224x3`). All 20 layers of the base model are frozen (`trainable = False`) to preserve feature representations.
- **Feature Flattening:** The `(7, 7, 512)` tensor from `block5_pool` is flattened into a 25,088-dimensional vector.
- **Classification Head:**
  - `Dense(512, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(512, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(1, activation='sigmoid')`
- **Total Parameters:** 27,823,425 (106.14 MB).
  - Trainable parameters: 13,108,737 (50.01 MB).
  - Non-trainable parameters: 14,714,688 (56.13 MB).
- **Compilation:** Adam optimizer ($\text{lr} = 0.0001$), Binary Crossentropy loss, metric `accuracy`.
- **Decision Rule:** Sigmoid output represents $P(\text{dent})$.
  $$\hat{y} = \begin{cases} \text{dent} (1) & \text{if } P(\text{dent}) > 0.5 \\ \text{crack} (0) & \text{if } P(\text{dent}) \le 0.5 \end{cases}$$
  $$P(\text{crack}) = 1.0 - P(\text{dent})$$

### 5.2 Module B: Vision-Language Model (BLIP)
- **Model Checkpoint:** `Salesforce/blip-image-captioning-base`.
- **Text Conditioning:**
  - **Caption Task:** Conditioned on the prompt prefix `"This is a picture of"`.
  - **Summary Task:** Conditioned on the prompt prefix `"This is a detailed photo showing"`.
- **Custom Keras Integration:** Subclassed `tf.keras.layers.Layer` (`BlipCaptionSummaryLayer`) invoking PyTorch inference via `tf.py_function(process_image, [image_path, task], tf.string)`.

---

## 6. Experimental Setup
- **Operating System:** Windows 11 Home Single Language (64-bit).
- **Environment:** Python 3.11.9 Virtual Environment (`.venv`).
- **Core Frameworks:** TensorFlow CPU 2.17.1, PyTorch CPU 2.2.0, Hugging Face Transformers 4.38.2, Keras 3.15.1, NumPy 1.26.4.
- **Reproducibility Seed:** 42 (`random.seed`, `np.random.seed`, `tf.random.set_seed`).
- **Hyperparameters:** Batch size = 32, Epochs = 5, Learning rate = 0.0001, Image resolution = 224x224 RGB, Scaling factor = 1./255.

---

## 7. Results & Analysis

### 7.1 Training Progression
Real metrics recorded across the 5 training epochs:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.7201 | 53.67% | 0.6338 | 60.42% |
| 2 | 0.5535 | 72.33% | 0.5821 | 69.79% |
| 3 | 0.4737 | 77.67% | 0.5486 | 71.88% |
| 4 | 0.4123 | 81.33% | 0.6457 | 65.62% |
| 5 | 0.3204 | 88.00% | 0.5098 | 70.83% |

- Final Training Accuracy: **88.00%** (Loss: 0.3204).
- Final Validation Accuracy: **70.83%** (Loss: 0.5098).
- Model persistence parity verified upon reloading: Maximum numerical difference = $0.00 \times 10^0$ ($\text{atol} = 10^{-5}$).

### 7.2 Test Split Evaluation
Dual evaluation was conducted:
1. **Academic Notebook Evaluation (Truncated Steps):**
   - Formula: `steps = test_generator.samples // test_generator.batch_size` = $50 // 32 = 1$ step.
   - Evaluated samples: 32 (18 trailing samples omitted due to integer truncation).
   - Test Loss: **0.7326**
   - Test Accuracy: **68.75%**
2. **Full Supplementary Evaluation (Complete Test Split):**
   - Evaluated samples: 50 / 50 (100% of test data).
   - Test Loss: **0.5783**
   - Test Accuracy: **76.00%**
   - **Confusion Matrix:**
     $$\begin{pmatrix} \text{TN (Crack)} = 15 & \text{FP (Dent)} = 10 \\ \text{FN (Crack)} = 2 & \text{TP (Dent)} = 23 \end{pmatrix}$$
   - **Positive Class (`dent`):** Precision = 0.6970, Recall = **0.9200**, F1-score = **0.7931**.
   - **Negative Class (`crack`):** Precision = 0.8824, Recall = 0.6000, F1-score = **0.7143**.

---

## 8. Discussion
1. **High Dent Recall (92.0%):** The model correctly identified 23 out of 25 dent images, demonstrating strong sensitivity for localized surface depressions.
2. **Crack False Positives (10 / 25):** 10 crack images were misclassified as dents. Qualitative inspection reveals that elongated structural lighting shadows on metallic fuselages can mimic dent contours.
3. **Small Sample Variance:** With 50 total test images, each image represents 2.0% accuracy. The difference between the 32-sample truncated formula (68.75%) and full evaluation (76.00%) underscores the critical need for complete un-truncated reporting.

---

## 9. Limitations & Ethical Considerations
1. **Summary Semantics:** In this lab, "summary" is not an extractive or abstractive text summarizer; it is the identical BLIP captioning checkpoint conditioned on an alternate prompt string.
2. **Aviation Safety Disclaimer:** The model is an academic demonstrator and is not certified for aviation maintenance sign-off.
3. **Keras Layer Serializability:** `BlipCaptionSummaryLayer` uses `tf.py_function` wrapping PyTorch code and cannot be serialized to a standard TensorFlow SavedModel graph.

---

## 10. References
1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.
2. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. *ICML 2022*.
3. Roboflow Universe. (2023). Aircraft Damage Detection Dataset (youssef-donia-fhktl). License: CC BY 4.0.
4. IBM Skills Network. (2025). Final Project: Classification and Captioning Aircraft Damage Using Pretrained Models.
