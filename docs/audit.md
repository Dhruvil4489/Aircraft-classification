# Repository and Environment Audit

**Audit Date:** 2026-10-17  
**Project:** Aircraft Damage Classification, Image Captioning and Summarization (Pretrained VGG16 + BLIP)  
**Workspace Root:** `D:\Aircraft classification`  

---

## 1. Ground Truth and Original Notebook Verification

- **Original Notebook File Location:** `notebooks/original/Final_Project_Classification_and_Captioning.ipynb`
- **Initial Location:** Root directory `Final_Project_Classification_and_Captioning.ipynb` (copied to `notebooks/original/`)
- **Calculated SHA-256:** `FF61DABC2DC3477C8A69B85815F9D25A695515D31C0B0C774603CC59B86FD7AC`
- **Integrity Policy:** The original notebook will remain strictly unmodified and byte-identical throughout all phases. Any modification will be rejected.

---

## 2. Environment and Hardware Inspection

| Property | Detected Value | Notes / Implications |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home Single Language (64-bit) | PowerShell shell execution |
| **Python Version** | Python 3.11.9 (`.venv\Scripts\python.exe`) | Clean venv created under `.venv` using Python 3.11.9 (compatible with TF 2.17.1 and PyTorch 2.2.0 CPU) |
| **Host System Pythons** | Python 3.14.4 (Default in AppData), Python 3.11.9 (AppData) | Pinned venv explicitly invokes `.\.venv\Scripts\python.exe` |
| **CPU / RAM** | 8 GB Physical Memory (Visible: ~7.7 GB, Free: ~0.7 GB - 1.2 GB dynamic) | Resource-conscious training and batch processing required |
| **GPU / Video Controller**| NVIDIA GeForce GTX 1650 / AMD Radeon Graphics | Native TensorFlow on Windows >= 2.11 is CPU-only. PyTorch CPU wheels used per notebook pin. CUDA will not be assumed. |
| **Storage (Free Space)**| Drive C: ~121 GB free; Drive D: ~84 GB free | Workspace on Drive D has ample space for models, datasets, and logs. |

---

## 3. Analysis of Original Notebook and Ambiguities

1. **Configuration Parameter Gaps in Cell `0abe19c5-b6ab-4a1e-bb61-fc018c1440a5`:**
   - The markdown cell states: *"Set the batch size is set to 32. Set the number of epcohs is 5."*
   - However, the code cell left `batch_size = ` and `n_epochs = ` blank without values.
   - **Resolution:** Explicitly set `batch_size = 32` and `n_epochs = 5` in config and the completed notebook.

2. **Colab-Specific Path in Task 9 Solution Cell `9ec0f0d9-f86d-41ad-9a24-be03a62f8fa5`:**
   - The original notebook contains a pre-filled solution snippet specifying: `image_path = "/content/sample_image.jpg"`.
   - On Windows / local execution, `/content/sample_image.jpg` does not exist and fails immediately.
   - **Resolution:** In the completed notebook, bind `image_path` to the real dataset image defined right above it: `aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg`. A markdown cell documenting this compatibility fix will be added.

3. **Data Download & Destruction Behavior in Cell `10b4107d-06ef-4d29-b4dd-f6ab9bb196e0`:**
   - The original notebook unconditionally removes `aircraft_damage_dataset_v1` using `shutil.rmtree` and re-downloads the 140MB archive on every run.
   - **Resolution:** In our preparation script and completed notebook, check if the dataset is already extracted and valid before deleting or re-downloading.

4. **Test Evaluation Step Truncation (`test_generator.samples // test_generator.batch_size`):**
   - In Cell `d677fc6a-032b-47af-8459-ec4dd5fe4f30`, `steps = test_generator.samples // test_generator.batch_size` is calculated.
   - If the test set has fewer images than the batch size (32), this integer division yields `steps = 0`, causing evaluation to skip or evaluate 0 samples. Even if samples >= 32, trailing images (samples % 32) are omitted.
   - **Resolution:** Retain the exact notebook formula for academic grading fidelity (P0), but add a comprehensive supplementary evaluation without step truncation that calculates accuracy, loss, confusion matrix, precision, recall, and F1 across all test samples.

5. **Keras 3 / TensorFlow 2.17 Compatibility for `BlipCaptionSummaryLayer`:**
   - In TensorFlow 2.17.1 (defaulting to Keras 3), `tf.keras.layers.Layer.call(self, image_path, task)` using `tf.py_function` must be validated to ensure proper tensor unpacking and string return types.
   - Non-trainability / serialization: BLIP wrapped via PyTorch inside a `tf.py_function` is non-differentiable and not exportable to SavedModel format. This limitation must be documented.

6. **OpenMP Runtime Coexistence (TF + PyTorch on Windows):**
   - Loading both TensorFlow and PyTorch within the same process on Windows can trigger OpenMP `libiomp5md.dll` initialization conflicts.
   - **Mitigation:** Lazily isolate imports, verify behavior, and if needed use thread/subprocess isolation or document `KMP_DUPLICATE_LIB_OK=TRUE` if strictly necessary.

---

## 4. Requirements Traceability Checklist (R01 – R31)

| Req ID | Category | Description | Priority | Target Verification | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R01** | Notebook | Original notebook inspected, preserved at `notebooks/original/`, SHA-256 recorded | P0 | Checksum verification | Verified |
| **R02** | Notebook | Completed notebook created at `notebooks/completed/` filling all 10 tasks | P0 | Notebook file audit | Pending Phase 7 |
| **R03** | Notebook | Batch size set to 32 and epochs set to 5 based on markdown specification | P0 | Code inspection | Pending Phase 7 |
| **R04** | Notebook | Compatibility change: Task 9 replaced `/content/sample_image.jpg` with valid dataset path | P0 | Notebook execution | Pending Phase 7 |
| **R05** | Notebook | Notebook executed end-to-end via nbclient/nbconvert with all real outputs saved | P0 | Execution log in artifacts/logs/ | Pending Phase 7 |
| **R06** | Notebook | Final cell logs exact environment versions (Python, TF, Keras, Torch, Transformers) | P0 | Notebook output | Pending Phase 7 |
| **R07** | Dataset | Dataset downloaded from official IBM Cloud Object Storage link and verified | P0 | `scripts/prepare_data.py` output | Pending Phase 3 |
| **R08** | Dataset | Roboflow "Aircraft Damage Detection" (youssef-donia, CC BY 4) attribution verified | P0 | README.md and UI | Pending Phase 3 |
| **R09** | Dataset | Safe archive extraction implemented (no path traversal, symlink checks) | P1 | Unit test | Pending Phase 3 |
| **R10** | Dataset | Dataset structure validated (train/valid/test, crack/dent), leakage and duplicate checks reported | P1 | `artifacts/results/dataset_report.json` | Pending Phase 3 |
| **R11** | Model | Class mapping verified and enforced: `{'crack': 0, 'dent': 1}` | P0 | Assertions & tests | Pending Phase 4 |
| **R12** | Model | Preprocessing strictly defined as RGB, resize to 224x224, scale 1./255 | P0 | `src/aircraft_ai/classifier.py` | Pending Phase 4 |
| **R13** | Model | Architecture: ImageNet VGG16 base (frozen, `include_top=False`), Flatten, Dense(512), Dropout(0.3), Dense(512), Dropout(0.3), Dense(1, 'sigmoid') | P0 | Layer summary inspect | Pending Phase 4 |
| **R14** | Model | Compile with Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'] | P0 | Model compile config | Pending Phase 4 |
| **R15** | Model | Train for 5 epochs on train_generator with valid_generator, save history to `artifacts/results/history.json` | P0 | Training execution | Pending Phase 4 |
| **R16** | Model | Generate training & validation loss curves and Task 6 accuracy curve (figsize 5x5, specified labels) | P0 | PNGs in artifacts/results/ | Pending Phase 4 |
| **R17** | Evaluation | Dual evaluation: notebook truncated steps formula + full test split metrics (confusion matrix, precision/recall/F1, loss, acc) | P0/P1 | `artifacts/results/metrics.json` | Pending Phase 4 |
| **R18** | Evaluation | Visualization: Task 7 `test_model_on_image` (index 1) and deterministic grid of test predictions | P0/P1 | Artifact plots | Pending Phase 4 |
| **R19** | Model | Model persistence: save `artifacts/models/vgg16_aircraft_damage.keras`, verify reload numerical parity | P1 | Model equality test | Pending Phase 4 |
| **R20** | BLIP | Load `Salesforce/blip-image-captioning-base` processor and conditional generation model | P0 | Model instantiation | Pending Phase 5 |
| **R21** | BLIP | Implement `describe_image(image, task)` for caption ("This is a picture of") and summary ("This is a detailed photo showing") | P0 | Core captioning module | Pending Phase 5 |
| **R22** | BLIP | Implement custom Keras layer `BlipCaptionSummaryLayer` and helper `generate_text(image_path, task)` matching notebook interface | P0 | `src/aircraft_ai/blip_layer.py` | Pending Phase 5 |
| **R23** | BLIP | Save real caption & summary examples for Task 9 & Task 10 images to `artifacts/results/blip_examples.json` | P0 | Real outputs JSON | Pending Phase 5 |
| **R24** | API | FastAPI service implemented in `src/aircraft_ai/api.py` with endpoints for classify, caption, summary, evaluation, info | P1 | API endpoint tests | Pending Phase 6 |
| **R25** | API | Lazy model singletons, concurrency semaphore for BLIP inference, error handling | P1 | Concurrency test | Pending Phase 6 |
| **R26** | Frontend | Responsive, accessible static web UI (`frontend/`) connecting to API endpoints without node build | P1 | Browser / HTTP smoke tests | Pending Phase 6 |
| **R27** | Security | File upload validation (types, max 10MB, PIL re-encoding, sanitized names) and allow-listed path traversal guards | P1 | Security tests | Pending Phase 6 |
| **R28** | Testing | Comprehensive test suite under `tests/` covering dataset, model, preprocessing, BLIP, API, and academic parameters | P1 | `pytest` test report | Pending Phase 8 |
| **R29** | Documentation | Complete README.md covering setup, execution, API docs, dataset license, limitations, and safety disclaimers | P1 | README file audit | Pending Phase 7 |
| **R30** | Documentation | Academic report outline in `docs/report_outline.md` filled with real metrics only | P0 | Report outline file | Pending Phase 7 |
| **R31** | Final Report | Complete verification report in `docs/completion_report.md` with evidence paths for every acceptance item | P0 | Completion report | Pending Phase 10 |

---

## 5. Identified Risks and Mitigation Strategies

1. **Risk:** Memory exhaustion during VGG16 training + BLIP loading on 8GB RAM host.  
   *Mitigation:* Sequence execution cleanly (train and evaluate VGG16 first, persist weights, release memory before executing heavy BLIP loads; in the API, load models lazily on first access).
2. **Risk:** OpenMP initialization crash when running TF and PyTorch in the same process.  
   *Mitigation:* Import TF and PyTorch carefully; test layer execution eagerly. If duplicate runtime initialization occurs, document and isolate or apply `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` with clear audit logging.
3. **Risk:** Network interruption during dataset or model weight download.  
   *Mitigation:* Implement retry logic, progress reporting, file caching, and fallback instructions for manual offline placement.
4. **Risk:** Small sample size in test split causing integer truncation or high variance.  
   *Mitigation:* Report exact sample counts per class, explain metric limitations in documentation, and provide full non-truncated test evaluation.
