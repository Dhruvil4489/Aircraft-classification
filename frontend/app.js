/**
 * AeroDamage AI — Frontend Application Logic
 * Vanilla JavaScript (ES6) for reactive, real inference and analytics.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let selectedClassifyBlob = null;
  let selectedCaptionBlob = null;
  let isClassifying = false;
  let isCaptioning = false;
  let isSummarizing = false;

  // DOM Elements - Navigation
  const navButtons = document.querySelectorAll(".nav-btn");
  const contentSections = document.querySelectorAll(".content-section");
  const statusIndicator = document.getElementById("system-status-indicator");
  const statusText = document.getElementById("system-status-text");
  const statusDot = statusIndicator?.querySelector(".status-dot");
  const alertBanner = document.getElementById("global-alert-banner");
  const alertMessage = document.getElementById("alert-message");
  const btnCloseAlert = document.getElementById("btn-close-alert");

  // Hero Actions
  document.getElementById("btn-hero-classify")?.addEventListener("click", () => switchSection("section-classify"));
  document.getElementById("btn-hero-evaluation")?.addEventListener("click", () => switchSection("section-evaluation"));

  // 1. Navigation Controller
  function switchSection(targetId) {
    contentSections.forEach((section) => {
      section.classList.toggle("active", section.id === targetId);
    });
    navButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.section === targetId);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.section) switchSection(btn.dataset.section);
    });
  });

  // Alert Handler
  function showAlert(msg, isWarning = true) {
    if (!alertBanner || !alertMessage) return;
    alertMessage.textContent = msg;
    alertBanner.style.display = "flex";
    alertBanner.style.background = isWarning ? "rgba(245, 158, 11, 0.15)" : "rgba(239, 68, 68, 0.15)";
    alertBanner.style.borderColor = isWarning ? "rgba(245, 158, 11, 0.4)" : "rgba(239, 68, 68, 0.4)";
  }

  btnCloseAlert?.addEventListener("click", () => {
    if (alertBanner) alertBanner.style.display = "none";
  });

  // 2. Health & Status Check
  async function checkSystemHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        if (statusDot) {
          statusDot.className = "status-dot online";
        }
        if (statusText) {
          statusText.textContent = data.model_trained ? "Model Ready (Trained)" : "API Online (Untrained)";
        }
        const badge = document.getElementById("model-status-badge");
        if (badge) {
          badge.textContent = data.model_trained ? "Model Ready" : "Untrained (Run Training)";
        }
      } else {
        throw new Error("Server responded with error status");
      }
    } catch (err) {
      if (statusDot) statusDot.className = "status-dot offline";
      if (statusText) statusText.textContent = "API Offline";
    }
  }

  // 3. Classification Setup
  const classifyDropzone = document.getElementById("classify-dropzone");
  const classifyFileInput = document.getElementById("classify-file-input");
  const classifyPreviewContainer = document.getElementById("classify-preview-container");
  const classifyPreviewImage = document.getElementById("classify-preview-image");
  const dropzoneContent = document.getElementById("dropzone-content");
  const btnRemoveClassifyPreview = document.getElementById("btn-remove-classify-preview");
  const btnRunClassify = document.getElementById("btn-run-classify");
  const classifySpinner = document.getElementById("classify-spinner");
  const classifyBtnText = document.getElementById("classify-btn-text");

  // Drag & drop
  classifyDropzone?.addEventListener("click", (e) => {
    if (e.target !== btnRemoveClassifyPreview) {
      classifyFileInput?.click();
    }
  });

  classifyDropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    classifyDropzone.classList.add("dragover");
  });

  classifyDropzone?.addEventListener("dragleave", () => {
    classifyDropzone.classList.remove("dragover");
  });

  classifyDropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    classifyDropzone.classList.remove("dragover");
    if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
      handleClassifyFile(e.dataTransfer.files[0]);
    }
  });

  classifyFileInput?.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleClassifyFile(e.target.files[0]);
    }
  });

  function handleClassifyFile(file) {
    if (!file.type.match(/image\/(jpeg|png|webp)/)) {
      showAlert("Unsupported file type. Please upload a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showAlert("File size exceeds 10 MB limit.");
      return;
    }
    selectedClassifyBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (classifyPreviewImage && classifyPreviewContainer && dropzoneContent) {
        classifyPreviewImage.src = e.target.result;
        classifyPreviewContainer.style.display = "flex";
        dropzoneContent.style.display = "none";
        btnRunClassify.disabled = false;
      }
    };
    reader.readAsDataURL(file);
  }

  btnRemoveClassifyPreview?.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedClassifyBlob = null;
    if (classifyPreviewContainer) classifyPreviewContainer.style.display = "none";
    if (dropzoneContent) dropzoneContent.style.display = "block";
    if (classifyFileInput) classifyFileInput.value = "";
    if (btnRunClassify) btnRunClassify.disabled = true;
    document.getElementById("classify-empty-state").style.display = "block";
    document.getElementById("classify-result-content").style.display = "none";
  });

  // Example buttons for classification
  document.querySelectorAll("#classify-example-buttons .example-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const fileName = btn.dataset.file;
      try {
        const res = await fetch(`/api/examples/${fileName}`);
        if (!res.ok) throw new Error("Example file not found on server.");
        const blob = await res.blob();
        handleClassifyFile(new File([blob], fileName, { type: "image/jpeg" }));
      } catch (err) {
        showAlert(`Could not load example image: ${err.message}`);
      }
    });
  });

  // Run Real VGG16 Classification
  btnRunClassify?.addEventListener("click", async () => {
    if (!selectedClassifyBlob || isClassifying) return;
    isClassifying = true;
    btnRunClassify.disabled = true;
    classifySpinner.style.display = "inline-block";
    classifyBtnText.textContent = "Running Inference...";

    const formData = new FormData();
    formData.append("file", selectedClassifyBlob);

    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      });

      if (res.status === 503) {
        const errData = await res.json();
        showAlert(errData.detail || "Model not trained yet. Please train the model first.", true);
        return;
      }

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Classification failed");
      }

      const data = await res.json();
      displayClassificationResult(data);
    } catch (err) {
      showAlert(`Classification Error: ${err.message}`);
    } finally {
      isClassifying = false;
      btnRunClassify.disabled = false;
      classifySpinner.style.display = "none";
      classifyBtnText.textContent = "Run VGG16 Classification";
    }
  });

  function displayClassificationResult(data) {
    document.getElementById("classify-empty-state").style.display = "none";
    const resBox = document.getElementById("classify-result-content");
    resBox.style.display = "block";

    const labelVal = document.getElementById("res-predicted-class");
    labelVal.textContent = data.predicted_class;
    labelVal.className = `prediction-label-value ${data.predicted_class}`;

    const probDentPct = (data.probability_dent * 100).toFixed(2);
    const probCrackPct = (data.probability_crack * 100).toFixed(2);

    document.getElementById("res-prob-dent").textContent = `${probDentPct}%`;
    document.getElementById("res-bar-dent").style.width = `${probDentPct}%`;

    document.getElementById("res-prob-crack").textContent = `${probCrackPct}%`;
    document.getElementById("res-bar-crack").style.width = `${probCrackPct}%`;

    document.getElementById("res-disclaimer").textContent = data.disclaimer;
  }

  // 4. Caption & Summary Setup
  const captionDropzone = document.getElementById("caption-dropzone");
  const captionFileInput = document.getElementById("caption-file-input");
  const captionPreviewContainer = document.getElementById("caption-preview-container");
  const captionPreviewImage = document.getElementById("caption-preview-image");
  const captionDropzoneContent = document.getElementById("caption-dropzone-content");
  const btnRemoveCaptionPreview = document.getElementById("btn-remove-caption-preview");
  const btnRunCaption = document.getElementById("btn-run-caption");
  const btnRunSummary = document.getElementById("btn-run-summary");
  const captionSpinner = document.getElementById("caption-spinner");
  const summarySpinner = document.getElementById("summary-spinner");

  captionDropzone?.addEventListener("click", (e) => {
    if (e.target !== btnRemoveCaptionPreview) {
      captionFileInput?.click();
    }
  });

  captionDropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    captionDropzone.classList.add("dragover");
  });

  captionDropzone?.addEventListener("dragleave", () => {
    captionDropzone.classList.remove("dragover");
  });

  captionDropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    captionDropzone.classList.remove("dragover");
    if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
      handleCaptionFile(e.dataTransfer.files[0]);
    }
  });

  captionFileInput?.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleCaptionFile(e.target.files[0]);
    }
  });

  function handleCaptionFile(file) {
    if (!file.type.match(/image\/(jpeg|png|webp)/)) {
      showAlert("Unsupported file type for captioning. Use JPEG, PNG or WebP.");
      return;
    }
    selectedCaptionBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      captionPreviewImage.src = e.target.result;
      captionPreviewContainer.style.display = "flex";
      captionDropzoneContent.style.display = "none";
      btnRunCaption.disabled = false;
      btnRunSummary.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  btnRemoveCaptionPreview?.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedCaptionBlob = null;
    captionPreviewContainer.style.display = "none";
    captionDropzoneContent.style.display = "block";
    captionFileInput.value = "";
    btnRunCaption.disabled = true;
    btnRunSummary.disabled = true;
    document.getElementById("caption-empty-state").style.display = "block";
    document.getElementById("caption-output-box").style.display = "none";
  });

  // Examples buttons for BLIP
  document.querySelectorAll("#caption-example-buttons .example-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const fileName = btn.dataset.file;
      try {
        const res = await fetch(`/api/examples/${fileName}`);
        if (!res.ok) throw new Error("Example file not found on server.");
        const blob = await res.blob();
        handleCaptionFile(new File([blob], fileName, { type: "image/jpeg" }));
      } catch (err) {
        showAlert(`Could not load example: ${err.message}`);
      }
    });
  });

  // Run BLIP Caption
  btnRunCaption?.addEventListener("click", async () => {
    if (!selectedCaptionBlob || isCaptioning) return;
    isCaptioning = true;
    btnRunCaption.disabled = true;
    captionSpinner.style.display = "inline-block";

    const formData = new FormData();
    formData.append("file", selectedCaptionBlob);

    try {
      const res = await fetch("/api/caption", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Captioning failed");
      }

      const data = await res.json();
      document.getElementById("caption-empty-state").style.display = "none";
      document.getElementById("caption-output-box").style.display = "flex";
      document.getElementById("blip-caption-text").textContent = `"${data.raw_generated_text}"`;
    } catch (err) {
      showAlert(`BLIP Caption Error: ${err.message}`);
    } finally {
      isCaptioning = false;
      btnRunCaption.disabled = false;
      captionSpinner.style.display = "none";
    }
  });

  // Run BLIP Summary
  btnRunSummary?.addEventListener("click", async () => {
    if (!selectedCaptionBlob || isSummarizing) return;
    isSummarizing = true;
    btnRunSummary.disabled = true;
    summarySpinner.style.display = "inline-block";

    const formData = new FormData();
    formData.append("file", selectedCaptionBlob);

    try {
      const res = await fetch("/api/summary", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Summarization failed");
      }

      const data = await res.json();
      document.getElementById("caption-empty-state").style.display = "none";
      document.getElementById("caption-output-box").style.display = "flex";
      document.getElementById("blip-summary-text").textContent = `"${data.raw_generated_text}"`;
    } catch (err) {
      showAlert(`BLIP Summary Error: ${err.message}`);
    } finally {
      isSummarizing = false;
      btnRunSummary.disabled = false;
      summarySpinner.style.display = "none";
    }
  });

  // 5. Load Evaluation & Metrics
  async function loadEvaluationData() {
    try {
      const res = await fetch("/api/evaluation");
      if (!res.ok) return;
      const data = await res.json();
      const full = data.full_supplementary_evaluation;
      if (full) {
        document.getElementById("kpi-full-acc").textContent = `${(full.test_accuracy * 100).toFixed(1)}%`;
        document.getElementById("kpi-full-loss").textContent = full.test_loss.toFixed(4);
        document.getElementById("kpi-full-samples").textContent = `Total Samples: ${full.full_samples_count}`;

        const dentM = full.positive_class_dent_metrics;
        if (dentM) {
          document.getElementById("kpi-dent-f1").textContent = dentM.f1_score.toFixed(4);
          document.getElementById("kpi-dent-prec-rec").textContent = `Prec: ${dentM.precision.toFixed(2)} | Rec: ${dentM.recall.toFixed(2)}`;
        }

        const crackM = full.negative_class_crack_metrics;
        if (crackM) {
          document.getElementById("kpi-crack-f1").textContent = crackM.f1_score.toFixed(4);
          document.getElementById("kpi-crack-prec-rec").textContent = `Prec: ${crackM.precision.toFixed(2)} | Rec: ${crackM.recall.toFixed(2)}`;
        }
      }
    } catch (err) {
      console.warn("Evaluation artifacts not yet generated:", err);
    }
  }

  // 6. Load Dataset & Leakage Info
  async function loadDatasetInfo() {
    try {
      const res = await fetch("/api/dataset/info");
      if (!res.ok) return;
      const data = await res.json();

      const tbody = document.getElementById("dataset-table-body");
      if (tbody && data.splits_summary) {
        tbody.innerHTML = "";
        for (const [split, counts] of Object.entries(data.splits_summary)) {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td><strong>${split.toUpperCase()}</strong></td>
            <td>${counts.crack}</td>
            <td>${counts.dent}</td>
            <td><strong>${counts.total}</strong></td>
          `;
          tbody.appendChild(row);
        }
      }

      if (data.leakage_analysis) {
        const baseCount = data.leakage_analysis.roboflow_base_id_overlaps_count ?? 0;
        const exactCount = data.leakage_analysis.cross_split_exact_duplicates_count ?? 0;
        document.getElementById("leakage-base-count").textContent = baseCount;
        document.getElementById("leakage-exact-count").textContent = exactCount;
      }
    } catch (err) {
      console.warn("Dataset report not yet generated:", err);
    }
  }

  // Initial runs
  checkSystemHealth();
  loadEvaluationData();
  loadDatasetInfo();
});
