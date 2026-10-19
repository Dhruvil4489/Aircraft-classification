/**
 * AeroDamage AI — Aerospace Engineering Workstation Client Application
 * Handles state management, multi-stage inspection telemetry, and API integration.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation elements
  const navButtons = document.querySelectorAll(".nav-btn");
  const contentSections = document.querySelectorAll(".content-section");
  const brandHome = document.getElementById("brand-home");
  const btnGotoAnalysis = document.getElementById("btn-goto-analysis");
  const btnGotoPerformance = document.getElementById("btn-goto-performance");

  // Alert elements
  const alertBanner = document.getElementById("global-alert-banner");
  const alertText = document.getElementById("global-alert-text");
  const btnCloseAlert = document.getElementById("btn-close-alert");

  // System status elements
  const statusDot = document.getElementById("system-status-dot");
  const statusLabel = document.getElementById("system-status-label");

  // Analysis elements
  const analysisDropzone = document.getElementById("analysis-dropzone");
  const analysisFileInput = document.getElementById("analysis-file-input");
  const analysisDropzoneContent = document.getElementById("analysis-dropzone-content");
  const analysisPreviewContainer = document.getElementById("analysis-preview-container");
  const analysisPreviewImg = document.getElementById("analysis-preview-img");
  const imgMetaDimensions = document.getElementById("img-meta-dimensions");
  const imgMetaSize = document.getElementById("img-meta-size");
  const imgMetaFormat = document.getElementById("img-meta-format");
  const btnReplaceAnalysisImg = document.getElementById("btn-replace-analysis-img");
  const btnRemoveAnalysisImg = document.getElementById("btn-remove-analysis-img");
  const btnRunAnalysis = document.getElementById("btn-run-analysis");
  const analysisSpinner = document.getElementById("analysis-spinner");
  const analysisBtnText = document.getElementById("analysis-btn-text");

  // Pipeline stage indicators
  const stepValidation = document.getElementById("step-validation");
  const stepScreening = document.getElementById("step-screening");
  const stepClassifier = document.getElementById("step-classifier");
  const stepUncertainty = document.getElementById("step-uncertainty");

  // Analysis result elements
  const analysisEmptyState = document.getElementById("analysis-empty-state");
  const analysisResultsContent = document.getElementById("analysis-results-content");
  const resDecisionBanner = document.getElementById("res-decision-banner");
  const resDecisionTitle = document.getElementById("res-decision-title");
  const resDecisionLabel = document.getElementById("res-decision-label");
  const resDecisionMessage = document.getElementById("res-decision-message");
  const resScoresBlock = document.getElementById("res-scores-block");
  const valScoreDent = document.getElementById("val-score-dent");
  const barScoreDent = document.getElementById("bar-score-dent");
  const valScoreCrack = document.getElementById("val-score-crack");
  const barScoreCrack = document.getElementById("bar-score-crack");
  const resDomainBlock = document.getElementById("res-domain-block");
  const valDomainScore = document.getElementById("val-domain-score");
  const barDomainScore = document.getElementById("bar-domain-score");
  const resMatchedCategory = document.getElementById("res-matched-category");
  const telPipelineStatus = document.getElementById("tel-pipeline-status");
  const telConfidenceType = document.getElementById("tel-confidence-type");
  const telDecisionMargin = document.getElementById("tel-decision-margin");
  const telPreprocessing = document.getElementById("tel-preprocessing");
  const resSafetyDisclaimer = document.getElementById("res-safety-disclaimer");

  // Captioning elements
  const captionDropzone = document.getElementById("caption-dropzone");
  const captionFileInput = document.getElementById("caption-file-input");
  const captionDropzoneContent = document.getElementById("caption-dropzone-content");
  const captionPreviewContainer = document.getElementById("caption-preview-container");
  const captionPreviewImg = document.getElementById("caption-preview-img");
  const btnRemoveCaptionImg = document.getElementById("btn-remove-caption-img");
  const btnRunCaption = document.getElementById("btn-run-caption");
  const btnRunSummary = document.getElementById("btn-run-summary");
  const captionSpinner = document.getElementById("caption-spinner");
  const summarySpinner = document.getElementById("summary-spinner");
  const captionEmptyState = document.getElementById("caption-empty-state");
  const captionResultsContent = document.getElementById("caption-results-content");
  const blipCaptionOutput = document.getElementById("blip-caption-output");
  const blipSummaryOutput = document.getElementById("blip-summary-output");

  // State
  let currentAnalysisBlob = null;
  let currentCaptionBlob = null;
  let isAnalyzing = false;

  // =========================================================================
  // Navigation & Tabs
  // =========================================================================
  function switchTab(sectionId) {
    navButtons.forEach(btn => {
      const isTarget = btn.dataset.section === sectionId;
      btn.classList.toggle("active", isTarget);
    });

    contentSections.forEach(sec => {
      sec.classList.toggle("active", sec.id === sectionId);
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navButtons.forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.section));
  });

  brandHome?.addEventListener("click", () => switchTab("section-overview"));
  btnGotoAnalysis?.addEventListener("click", () => switchTab("section-analysis"));
  btnGotoPerformance?.addEventListener("click", () => switchTab("section-performance"));

  // =========================================================================
  // Alert Notifications
  // =========================================================================
  function showAlert(msg, type = "amber") {
    if (!alertBanner || !alertText) return;
    alertBanner.className = `alert-banner ${type}`;
    alertText.textContent = msg;
    alertBanner.style.display = "flex";
  }

  function hideAlert() {
    if (alertBanner) alertBanner.style.display = "none";
  }

  btnCloseAlert?.addEventListener("click", hideAlert);

  // =========================================================================
  // Live Telemetry Initialization
  // =========================================================================
  async function initSystemTelemetry() {
    try {
      const res = await fetch("/api/model/status");
      if (!res.ok) throw new Error("Status API unreachable");
      const data = await res.json();

      if (data.model_trained) {
        statusDot?.classList.add("online");
        statusDot?.classList.remove("offline");
        if (statusLabel) statusLabel.textContent = "Telemetry Online — VGG16 & CLIP Active";
        const kpiClassifier = document.getElementById("kpi-classifier-status");
        if (kpiClassifier) kpiClassifier.textContent = "Ready (v1.0)";
      } else {
        statusDot?.classList.add("offline");
        statusDot?.classList.remove("online");
        if (statusLabel) statusLabel.textContent = "Model Artifact Absent";
      }
    } catch (err) {
      statusDot?.classList.add("offline");
      if (statusLabel) statusLabel.textContent = "Telemetry Disconnected";
    }

    // Load evaluation metrics
    try {
      const evalRes = await fetch("/api/evaluation");
      if (evalRes.ok) {
        const evalData = await evalRes.json();
        const full = evalData.full_supplementary_evaluation || {};
        
        const kpiAcc = document.getElementById("kpi-eval-acc");
        if (kpiAcc && full.test_accuracy !== undefined) {
          kpiAcc.textContent = `${(full.test_accuracy * 100).toFixed(1)}%`;
        }

        const kpiLoss = document.getElementById("kpi-eval-loss");
        if (kpiLoss && full.test_loss !== undefined) {
          kpiLoss.textContent = full.test_loss.toFixed(4);
        }

        const kpiDentF1 = document.getElementById("kpi-eval-dent-f1");
        const dentMetrics = full.positive_class_dent_metrics || {};
        if (kpiDentF1 && dentMetrics.f1_score !== undefined) {
          kpiDentF1.textContent = `${(dentMetrics.f1_score * 100).toFixed(1)}%`;
        }

        const kpiCrackF1 = document.getElementById("kpi-eval-crack-f1");
        const crackMetrics = full.negative_class_crack_metrics || {};
        if (kpiCrackF1 && crackMetrics.f1_score !== undefined) {
          kpiCrackF1.textContent = `${(crackMetrics.f1_score * 100).toFixed(1)}%`;
        }
      }
    } catch (e) {
      console.warn("Evaluation metrics fetch failed:", e);
    }
  }

  initSystemTelemetry();

  // =========================================================================
  // Damage Analysis Viewport Handlers
  // =========================================================================
  analysisDropzone?.addEventListener("click", (e) => {
    if (!analysisPreviewContainer || analysisPreviewContainer.style.display === "none") {
      analysisFileInput?.click();
    }
  });

  btnReplaceAnalysisImg?.addEventListener("click", (e) => {
    e.stopPropagation();
    analysisFileInput?.click();
  });

  analysisDropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    analysisDropzone.classList.add("dragover");
  });

  analysisDropzone?.addEventListener("dragleave", () => {
    analysisDropzone.classList.remove("dragover");
  });

  analysisDropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    analysisDropzone.classList.remove("dragover");
    if (e.dataTransfer?.files?.[0]) {
      handleAnalysisFile(e.dataTransfer.files[0]);
    }
  });

  analysisFileInput?.addEventListener("change", (e) => {
    if (e.target.files?.[0]) {
      handleAnalysisFile(e.target.files[0]);
    }
  });

  function handleAnalysisFile(file) {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/i) && !file.name.match(/\.(jpe?g|png|webp)$/i)) {
      showAlert("Unsupported file format. Please upload a valid JPEG, PNG, or WebP image.", "error");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      showAlert("Upload exceeds 10 MB limit. Please select a smaller file.", "error");
      return;
    }

    hideAlert();
    currentAnalysisBlob = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (analysisPreviewImg && analysisPreviewContainer && analysisDropzoneContent) {
        analysisPreviewImg.src = e.target.result;
        
        // Populate natural dimensions once loaded
        analysisPreviewImg.onload = () => {
          if (imgMetaDimensions) {
            imgMetaDimensions.textContent = `${analysisPreviewImg.naturalWidth} \u00D7 ${analysisPreviewImg.naturalHeight} px`;
          }
        };

        if (imgMetaSize) {
          const kb = (file.size / 1024).toFixed(1);
          imgMetaSize.textContent = `${kb} KB`;
        }

        if (imgMetaFormat) {
          imgMetaFormat.textContent = file.type ? file.type.replace("image/", "").toUpperCase() : "IMAGE";
        }

        analysisPreviewContainer.style.display = "flex";
        analysisDropzoneContent.style.display = "none";
        btnRunAnalysis.disabled = false;
        resetPipelineStages();
      }
    };
    reader.readAsDataURL(file);
  }

  btnRemoveAnalysisImg?.addEventListener("click", (e) => {
    e.stopPropagation();
    currentAnalysisBlob = null;
    if (analysisPreviewContainer) analysisPreviewContainer.style.display = "none";
    if (analysisDropzoneContent) analysisDropzoneContent.style.display = "block";
    if (analysisFileInput) analysisFileInput.value = "";
    if (btnRunAnalysis) btnRunAnalysis.disabled = true;
    resetPipelineStages();
    if (analysisEmptyState) analysisEmptyState.style.display = "block";
    if (analysisResultsContent) analysisResultsContent.style.display = "none";
  });

  function resetPipelineStages() {
    [stepValidation, stepScreening, stepClassifier, stepUncertainty].forEach(step => {
      if (step) {
        step.className = "stage-step";
      }
    });
  }

  // Pre-verified Benchmark Sample Buttons
  document.querySelectorAll(".example-btn[data-file]").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const fileName = btn.dataset.file;
      try {
        const res = await fetch(`/api/examples/${fileName}`);
        if (!res.ok) throw new Error("Sample file not found on server");
        const blob = await res.blob();
        handleAnalysisFile(new File([blob], fileName, { type: "image/jpeg" }));
      } catch (err) {
        showAlert(`Failed to load benchmark sample: ${err.message}`, "error");
      }
    });
  });

  // Out-of-Domain Signature Fixture Generator Button
  const btnSampleSignature = document.getElementById("btn-sample-signature");
  btnSampleSignature?.addEventListener("click", (e) => {
    e.stopPropagation();
    // Render high-contrast synthetic handwritten signature fixture on canvas
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 200;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 400, 200);

    // Draw cursive ink strokes
    ctx.strokeStyle = "#001a70";
    ctx.lineWidth = 3.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(35, 110);
    ctx.bezierCurveTo(70, 30, 90, 160, 130, 70);
    ctx.bezierCurveTo(160, 20, 180, 140, 220, 60);
    ctx.bezierCurveTo(250, 110, 290, 40, 340, 140);
    ctx.stroke();

    // Underline stroke
    ctx.beginPath();
    ctx.moveTo(50, 150);
    ctx.lineTo(360, 138);
    ctx.stroke();

    canvas.toBlob((blob) => {
      handleAnalysisFile(new File([blob], "handwritten_signature_ood_fixture.png", { type: "image/png" }));
    }, "image/png");
  });

  // =========================================================================
  // Execute 4-Stage Inspection Pipeline
  // =========================================================================
  btnRunAnalysis?.addEventListener("click", async () => {
    if (!currentAnalysisBlob || isAnalyzing) return;
    isAnalyzing = true;
    btnRunAnalysis.disabled = true;
    if (analysisSpinner) analysisSpinner.style.display = "inline-block";
    if (analysisBtnText) analysisBtnText.textContent = "Executing 4-Stage Pipeline...";

    // Stage 1: File check active
    resetPipelineStages();
    if (stepValidation) stepValidation.className = "stage-step active";

    const formData = new FormData();
    formData.append("file", currentAnalysisBlob);

    try {
      // Transition to Stage 2: Domain Screening
      if (stepValidation) stepValidation.className = "stage-step success";
      if (stepScreening) stepScreening.className = "stage-step active";

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Analysis request failed");
      }

      const data = await res.json();
      displayAnalysisResult(data);
    } catch (err) {
      showAlert(`Pipeline Execution Failure: ${err.message}`, "error");
      if (stepScreening) stepScreening.className = "stage-step rejected";
    } finally {
      isAnalyzing = false;
      btnRunAnalysis.disabled = false;
      if (analysisSpinner) analysisSpinner.style.display = "none";
      if (analysisBtnText) analysisBtnText.textContent = "Execute 4-Stage Inspection Pipeline";
    }
  });

  function displayAnalysisResult(data) {
    if (analysisEmptyState) analysisEmptyState.style.display = "none";
    if (analysisResultsContent) analysisResultsContent.style.display = "block";

    const isSupported = data.is_supported;
    const status = data.status;

    if (!isSupported) {
      // Stage 2 rejected: Non-aircraft image
      if (stepValidation) stepValidation.className = "stage-step success";
      if (stepScreening) stepScreening.className = "stage-step rejected";
      if (stepClassifier) stepClassifier.className = "stage-step";
      if (stepUncertainty) stepUncertainty.className = "stage-step";

      // Banner styling
      if (resDecisionBanner) resDecisionBanner.className = "decision-banner unsupported";
      if (resDecisionTitle) resDecisionTitle.textContent = "DOMAIN REJECTION — OUT-OF-DISTRIBUTION";
      if (resDecisionLabel) resDecisionLabel.textContent = "UNSUPPORTED IMAGE";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      // Hide VGG16 score bars
      if (resScoresBlock) resScoresBlock.style.display = "none";

      // Show Domain screening block
      if (resDomainBlock) {
        resDomainBlock.style.display = "block";
        const oodPct = (data.confidence * 100).toFixed(1);
        if (valDomainScore) valDomainScore.textContent = `${(data.domain_screening.in_domain_score * 100).toFixed(1)}% In-Domain (${oodPct}% Reject Confidence)`;
        if (barDomainScore) barDomainScore.style.width = `${(data.domain_screening.in_domain_score * 100).toFixed(1)}%`;
        if (resMatchedCategory) resMatchedCategory.textContent = data.domain_screening.matched_category || "Non-Aircraft Image";
      }

      // Telemetry items
      if (telPipelineStatus) telPipelineStatus.textContent = "REJECTED (OOD)";
      if (telConfidenceType) telConfidenceType.textContent = "Zero-Shot CLIP Screening";
      if (telDecisionMargin) telDecisionMargin.textContent = "N/A (Classifier Not Run)";
      if (telPreprocessing) telPreprocessing.textContent = "Screened Before VGG16 Input";
    } else if (status === "UNCERTAIN") {
      // Stage 4 uncertain
      if (stepValidation) stepValidation.className = "stage-step success";
      if (stepScreening) stepScreening.className = "stage-step success";
      if (stepClassifier) stepClassifier.className = "stage-step success";
      if (stepUncertainty) stepUncertainty.className = "stage-step rejected";

      if (resDecisionBanner) resDecisionBanner.className = "decision-banner uncertain";
      if (resDecisionTitle) resDecisionTitle.textContent = "AMBIGUOUS STRUCTURAL DEFECT";
      if (resDecisionLabel) resDecisionLabel.textContent = "UNCERTAIN — MANUAL REVIEW REQUIRED";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      renderScoreBars(data);
    } else {
      // Supported and classified
      if (stepValidation) stepValidation.className = "stage-step success";
      if (stepScreening) stepScreening.className = "stage-step success";
      if (stepClassifier) stepClassifier.className = "stage-step success";
      if (stepUncertainty) stepUncertainty.className = "stage-step success";

      const isDent = data.predicted_class === "dent";
      if (resDecisionBanner) resDecisionBanner.className = `decision-banner ${isDent ? "dent" : "crack"}`;
      if (resDecisionTitle) resDecisionTitle.textContent = "STRUCTURAL DEFECT CONFIRMED";
      if (resDecisionLabel) resDecisionLabel.textContent = isDent ? "SURFACE DENT" : "FATIGUE CRACK";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      renderScoreBars(data);
    }

    if (resSafetyDisclaimer) resSafetyDisclaimer.textContent = data.disclaimer;
  }

  function renderScoreBars(data) {
    if (resScoresBlock) resScoresBlock.style.display = "block";
    if (resDomainBlock) resDomainBlock.style.display = "none";

    const dentPct = (data.raw_dent_score * 100).toFixed(2);
    const crackPct = (data.raw_crack_score * 100).toFixed(2);

    if (valScoreDent) valScoreDent.textContent = `${dentPct}%`;
    if (barScoreDent) barScoreDent.style.width = `${dentPct}%`;

    if (valScoreCrack) valScoreCrack.textContent = `${crackPct}%`;
    if (barScoreCrack) barScoreCrack.style.width = `${crackPct}%`;

    if (telPipelineStatus) telPipelineStatus.textContent = "VERIFIED";
    if (telConfidenceType) telConfidenceType.textContent = "Raw Sigmoid Activation";
    if (telDecisionMargin) telDecisionMargin.textContent = `\u00B1 ${data.margin_from_decision?.toFixed(4) || "0.0000"}`;
    if (telPreprocessing) telPreprocessing.textContent = "224\u00D7224 RGB, 1./255";
  }

  // =========================================================================
  // Module B: Captioning & Summary (BLIP)
  // =========================================================================
  captionDropzone?.addEventListener("click", () => {
    if (!captionPreviewContainer || captionPreviewContainer.style.display === "none") {
      captionFileInput?.click();
    }
  });

  captionFileInput?.addEventListener("change", (e) => {
    if (e.target.files?.[0]) handleCaptionFile(e.target.files[0]);
  });

  function handleCaptionFile(file) {
    currentCaptionBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (captionPreviewImg && captionPreviewContainer && captionDropzoneContent) {
        captionPreviewImg.src = e.target.result;
        captionPreviewContainer.style.display = "flex";
        captionDropzoneContent.style.display = "none";
        btnRunCaption.disabled = false;
        btnRunSummary.disabled = false;
      }
    };
    reader.readAsDataURL(file);
  }

  btnRemoveCaptionImg?.addEventListener("click", (e) => {
    e.stopPropagation();
    currentCaptionBlob = null;
    if (captionPreviewContainer) captionPreviewContainer.style.display = "none";
    if (captionDropzoneContent) captionDropzoneContent.style.display = "block";
    if (captionFileInput) captionFileInput.value = "";
    if (btnRunCaption) btnRunCaption.disabled = true;
    if (btnRunSummary) btnRunSummary.disabled = true;
    if (captionEmptyState) captionEmptyState.style.display = "block";
    if (captionResultsContent) captionResultsContent.style.display = "none";
  });

  document.querySelectorAll(".example-buttons .example-btn[id^='btn-caption-sample']").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const fileName = btn.dataset.file;
      try {
        const res = await fetch(`/api/examples/${fileName}`);
        if (!res.ok) throw new Error("Sample file not found on server");
        const blob = await res.blob();
        handleCaptionFile(new File([blob], fileName, { type: "image/jpeg" }));
      } catch (err) {
        showAlert(`Failed to load BLIP benchmark image: ${err.message}`, "error");
      }
    });
  });

  btnRunCaption?.addEventListener("click", async () => {
    if (!currentCaptionBlob) return;
    btnRunCaption.disabled = true;
    if (captionSpinner) captionSpinner.style.display = "inline-block";
    if (captionEmptyState) captionEmptyState.style.display = "none";
    if (captionResultsContent) captionResultsContent.style.display = "block";

    const formData = new FormData();
    formData.append("file", currentCaptionBlob);

    try {
      const res = await fetch("/api/caption", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Caption generation request failed");
      const data = await res.json();
      if (blipCaptionOutput) blipCaptionOutput.textContent = `"${data.raw_generated_text}"`;
    } catch (err) {
      if (blipCaptionOutput) blipCaptionOutput.textContent = `Error: ${err.message}`;
    } finally {
      btnRunCaption.disabled = false;
      if (captionSpinner) captionSpinner.style.display = "none";
    }
  });

  btnRunSummary?.addEventListener("click", async () => {
    if (!currentCaptionBlob) return;
    btnRunSummary.disabled = true;
    if (summarySpinner) summarySpinner.style.display = "inline-block";
    if (captionEmptyState) captionEmptyState.style.display = "none";
    if (captionResultsContent) captionResultsContent.style.display = "block";

    const formData = new FormData();
    formData.append("file", currentCaptionBlob);

    try {
      const res = await fetch("/api/summary", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Summary generation request failed");
      const data = await res.json();
      if (blipSummaryOutput) blipSummaryOutput.textContent = `"${data.raw_generated_text}"`;
    } catch (err) {
      if (blipSummaryOutput) blipSummaryOutput.textContent = `Error: ${err.message}`;
    } finally {
      btnRunSummary.disabled = false;
      if (summarySpinner) summarySpinner.style.display = "none";
    }
  });
});
