/**
 * AeroDamage AI — Aerospace Engineering Workstation Client Application
 * Handles state management, Grad-CAM visual evidence inspection,
 * operational defect reporting, and API integration.
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

  // Evidence View Switcher elements
  const viewModeButtons = document.querySelectorAll(".view-mode-btn");
  const btnViewOrig = document.getElementById("btn-view-orig");
  const btnViewOverlay = document.getElementById("btn-view-overlay");
  const btnViewHeatmap = document.getElementById("btn-view-heatmap");
  const overlaySliderRow = document.getElementById("overlay-slider-row");
  const overlayOpacitySlider = document.getElementById("overlay-opacity");
  const opacityValLabel = document.getElementById("opacity-val");
  const viewModeCaption = document.getElementById("view-mode-caption");

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
  const resEvidenceCard = document.getElementById("res-evidence-card");
  const resEvidenceText = document.getElementById("res-evidence-text");
  const resActionBox = document.getElementById("res-action-box");
  const resRecommendedAction = document.getElementById("res-recommended-action");
  const btnExportReport = document.getElementById("btn-export-report");
  const btnExportJson = document.getElementById("btn-export-json");

  // Telemetry items inside collapsible details
  const telInspectionId = document.getElementById("tel-inspection-id");
  const telProcessingTime = document.getElementById("tel-processing-time");
  const telAircraftStatus = document.getElementById("tel-aircraft-status");
  const telPresenceStatus = document.getElementById("tel-presence-status");
  const telDamageType = document.getElementById("tel-damage-type");
  const telUncertaintyStatus = document.getElementById("tel-uncertainty-status");
  const telDecisionMargin = document.getElementById("tel-decision-margin");
  const telModelVersion = document.getElementById("tel-model-version");
  const resSafetyDisclaimer = document.getElementById("res-safety-disclaimer");

  // History elements
  const historyTbody = document.getElementById("history-tbody");
  const historyCountBadge = document.getElementById("history-count-badge");

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

  // Application State
  let currentAnalysisBlob = null;
  let currentAnalysisFilename = "";
  let currentOriginalDataUrl = "";
  let currentOverlayDataUrl = "";
  let currentHeatmapDataUrl = "";
  let currentInspectionResult = null;
  let currentCaptionBlob = null;
  let isAnalyzing = false;
  let inspectionHistory = [];

  try {
    const stored = sessionStorage.getItem("aerodamage_inspections");
    if (stored) {
      inspectionHistory = JSON.parse(stored);
    }
  } catch (e) {
    inspectionHistory = [];
  }

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
  // System Telemetry & Evaluation Initialization
  // =========================================================================
  async function initSystemTelemetry() {
    try {
      const res = await fetch("/api/model/status");
      if (!res.ok) throw new Error("Status API unreachable");
      const data = await res.json();

      if (data.model_trained) {
        statusDot?.classList.add("online");
        statusDot?.classList.remove("offline");
        if (statusLabel) statusLabel.textContent = "System Operational";
        const kpiClassifier = document.getElementById("kpi-classifier-status");
        if (kpiClassifier) kpiClassifier.textContent = "Ready (v1.0)";
      } else {
        statusDot?.classList.add("offline");
        statusDot?.classList.remove("online");
        if (statusLabel) statusLabel.textContent = "Model Artifact Absent";
      }
    } catch (err) {
      statusDot?.classList.add("offline");
      if (statusLabel) statusLabel.textContent = "Service Disconnected";
    }

    // Load evaluation metrics into Overview & Performance tabs
    try {
      const evalRes = await fetch("/api/evaluation");
      if (evalRes.ok) {
        const evalData = await evalRes.json();
        const full = evalData.full_supplementary_evaluation || {};

        const kpiAcc = document.getElementById("kpi-eval-acc");
        if (kpiAcc && full.test_accuracy !== undefined) {
          kpiAcc.textContent = `${(full.test_accuracy * 100).toFixed(1)}%`;
        }

        const kpiOverviewAcc = document.getElementById("kpi-test-accuracy");
        if (kpiOverviewAcc && full.test_accuracy !== undefined) {
          kpiOverviewAcc.textContent = `${(full.test_accuracy * 100).toFixed(1)}%`;
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

    renderHistoryTable();
  }

  initSystemTelemetry();

  // =========================================================================
  // Damage Analysis Viewport Handlers
  // =========================================================================
  analysisDropzone?.addEventListener("click", () => {
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

  function clearAnalysisResult() {
    currentInspectionResult = null;
    currentOverlayDataUrl = "";
    currentHeatmapDataUrl = "";
    if (analysisEmptyState) analysisEmptyState.style.display = "block";
    if (analysisResultsContent) analysisResultsContent.style.display = "none";
    if (btnViewOverlay) btnViewOverlay.disabled = true;
    if (btnViewHeatmap) btnViewHeatmap.disabled = true;
    if (overlaySliderRow) overlaySliderRow.style.display = "none";
    setActiveViewMode("original");
  }

  function handleAnalysisFile(file) {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/i) && !file.name.match(/\.(jpe?g|png|webp)$/i)) {
      showAlert("Unsupported file format. Please upload a valid JPEG, PNG, or WebP image.", "crimson");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      showAlert("Upload exceeds 10 MB limit. Please select a smaller file.", "crimson");
      return;
    }

    hideAlert();
    clearAnalysisResult();

    currentAnalysisBlob = file;
    currentAnalysisFilename = file.name || "inspection_image.jpg";

    const reader = new FileReader();
    reader.onload = (e) => {
      currentOriginalDataUrl = e.target.result;
      if (analysisPreviewImg && analysisPreviewContainer && analysisDropzoneContent) {
        analysisPreviewImg.src = currentOriginalDataUrl;
        analysisPreviewImg.style.opacity = "1";

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
        if (viewModeCaption) viewModeCaption.textContent = "Original inspection panel photograph.";
      }
    };
    reader.readAsDataURL(file);
  }

  btnRemoveAnalysisImg?.addEventListener("click", (e) => {
    e.stopPropagation();
    currentAnalysisBlob = null;
    currentAnalysisFilename = "";
    currentOriginalDataUrl = "";
    if (analysisPreviewContainer) analysisPreviewContainer.style.display = "none";
    if (analysisDropzoneContent) analysisDropzoneContent.style.display = "block";
    if (analysisFileInput) analysisFileInput.value = "";
    if (btnRunAnalysis) btnRunAnalysis.disabled = true;
    clearAnalysisResult();
  });

  // Benchmark Buttons (Test split images)
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
        showAlert(`Failed to load benchmark sample: ${err.message}`, "crimson");
      }
    });
  });

  // =========================================================================
  // Evidence View Switcher (Original / Attention Overlay / Grad-CAM Map)
  // =========================================================================
  function setActiveViewMode(mode) {
    viewModeButtons.forEach(btn => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });

    if (!analysisPreviewImg) return;

    if (mode === "original") {
      analysisPreviewImg.src = currentOriginalDataUrl;
      analysisPreviewImg.style.opacity = "1";
      if (overlaySliderRow) overlaySliderRow.style.display = "none";
      if (viewModeCaption) viewModeCaption.textContent = "Original high-resolution panel photograph.";
    } else if (mode === "overlay") {
      analysisPreviewImg.src = currentOverlayDataUrl || currentOriginalDataUrl;
      const opacity = (parseInt(overlayOpacitySlider?.value || "45", 10) / 100);
      analysisPreviewImg.style.opacity = "1";
      if (overlaySliderRow) overlaySliderRow.style.display = "flex";
      if (viewModeCaption) {
        viewModeCaption.textContent = "Blended explainability overlay (Grad-CAM attention mapped onto panel surface).";
      }
    } else if (mode === "heatmap") {
      analysisPreviewImg.src = currentHeatmapDataUrl || currentOriginalDataUrl;
      analysisPreviewImg.style.opacity = "1";
      if (overlaySliderRow) overlaySliderRow.style.display = "none";
      if (viewModeCaption) {
        viewModeCaption.textContent = "Grad-CAM attention activation heatmap (block5_conv3 convolutional features).";
      }
    }
  }

  btnViewOrig?.addEventListener("click", () => setActiveViewMode("original"));
  btnViewOverlay?.addEventListener("click", () => setActiveViewMode("overlay"));
  btnViewHeatmap?.addEventListener("click", () => setActiveViewMode("heatmap"));

  overlayOpacitySlider?.addEventListener("input", (e) => {
    const val = e.target.value;
    if (opacityValLabel) opacityValLabel.textContent = `${val}%`;
  });

  // =========================================================================
  // Execute Inspection Pipeline
  // =========================================================================
  btnRunAnalysis?.addEventListener("click", async () => {
    if (!currentAnalysisBlob || isAnalyzing) return;
    isAnalyzing = true;
    btnRunAnalysis.disabled = true;
    if (analysisSpinner) analysisSpinner.style.display = "inline-block";
    if (analysisBtnText) analysisBtnText.textContent = "Analyzing Panel Image...";

    const formData = new FormData();
    formData.append("file", currentAnalysisBlob);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Analysis request failed");
      }

      const data = await res.json();
      currentInspectionResult = data;
      displayAnalysisResult(data);
      saveInspectionToHistory(data);
    } catch (err) {
      showAlert(`Inspection Pipeline Failure: ${err.message}`, "crimson");
    } finally {
      isAnalyzing = false;
      btnRunAnalysis.disabled = false;
      if (analysisSpinner) analysisSpinner.style.display = "none";
      if (analysisBtnText) analysisBtnText.textContent = "Analyze Inspection Image";
    }
  });

  function displayAnalysisResult(data) {
    if (analysisEmptyState) analysisEmptyState.style.display = "none";
    if (analysisResultsContent) analysisResultsContent.style.display = "block";

    const isSupported = data.is_supported;
    const status = data.status;
    const aircraftStatus = data.aircraft_image_status || (isSupported ? "SUPPORTED" : "UNSUPPORTED");
    const presenceStatus = data.damage_presence_status || "NOT_ASSESSED";
    const damageType = data.damage_type || data.predicted_class;

    // Handle Grad-CAM visual evidence
    const loc = data.localization || {};
    if (loc.available && loc.overlay_data_url && loc.heatmap_data_url) {
      currentOverlayDataUrl = loc.overlay_data_url;
      currentHeatmapDataUrl = loc.heatmap_data_url;
      if (btnViewOverlay) btnViewOverlay.disabled = false;
      if (btnViewHeatmap) btnViewHeatmap.disabled = false;
      setActiveViewMode("overlay");

      if (resEvidenceCard) resEvidenceCard.style.display = "block";
      if (resEvidenceText) {
        resEvidenceText.textContent = `Model attention concentrated in ${loc.focus_region || "panel surface region"}. Mean activation intensity: ${loc.mean_activation || "N/A"}.`;
      }
    } else {
      if (btnViewOverlay) btnViewOverlay.disabled = true;
      if (btnViewHeatmap) btnViewHeatmap.disabled = true;
      if (resEvidenceCard) resEvidenceCard.style.display = "none";
      setActiveViewMode("original");
    }

    if (!isSupported || aircraftStatus === "UNSUPPORTED") {
      // Stage A: Out-of-Domain Image
      if (resDecisionBanner) resDecisionBanner.className = "decision-banner unsupported";
      if (resDecisionTitle) resDecisionTitle.textContent = "STAGE A: DOMAIN SCREENING REJECTION";
      if (resDecisionLabel) resDecisionLabel.textContent = "UNSUPPORTED IMAGE";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      if (resScoresBlock) resScoresBlock.style.display = "none";

      if (resActionBox) {
        resActionBox.className = "action-recommendation-box";
        if (resRecommendedAction) resRecommendedAction.textContent = data.recommended_action;
      }
    } else if (status === "DAMAGE_PRESENCE_UNVERIFIED" || presenceStatus === "NOT_ASSESSED") {
      // Stage B: Ordinary Aircraft Image Accepted — Damage Presence Unverified
      if (resDecisionBanner) resDecisionBanner.className = "decision-banner unverified";
      if (resDecisionTitle) resDecisionTitle.textContent = "STAGE B: DAMAGE PRESENCE UNVERIFIED";
      if (resDecisionLabel) resDecisionLabel.textContent = "NO VERIFIED DAMAGE FINDING";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      // Crucial: Suppress score bars for ordinary aircraft
      if (resScoresBlock) resScoresBlock.style.display = "none";

      if (resActionBox) {
        resActionBox.className = "action-recommendation-box";
        if (resRecommendedAction) resRecommendedAction.textContent = data.recommended_action;
      }
    } else if (status === "UNCERTAIN" || damageType === "UNCERTAIN") {
      // Stage C: Damage Detected but Ambiguous Defect Type
      if (resDecisionBanner) resDecisionBanner.className = "decision-banner uncertain";
      if (resDecisionTitle) resDecisionTitle.textContent = "STAGE C: AMBIGUOUS STRUCTURAL EVIDENCE";
      if (resDecisionLabel) resDecisionLabel.textContent = "UNCERTAIN — MANUAL REVIEW REQUIRED";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      renderScoreBars(data);

      if (resActionBox) {
        resActionBox.className = "action-recommendation-box";
        if (resRecommendedAction) resRecommendedAction.textContent = data.recommended_action;
      }
    } else {
      // Stage C: Confident Damage Type Classification
      const isDent = damageType === "dent";
      if (resDecisionBanner) resDecisionBanner.className = `decision-banner ${isDent ? "dent" : "crack"}`;
      if (resDecisionTitle) resDecisionTitle.textContent = isDent ? "STAGE C: SURFACE DENT IDENTIFIED" : "STAGE C: FATIGUE CRACK IDENTIFIED";
      if (resDecisionLabel) resDecisionLabel.textContent = isDent ? "SURFACE DENT DETECTED" : "FATIGUE CRACK DETECTED";
      if (resDecisionMessage) resDecisionMessage.textContent = data.decision_message;

      renderScoreBars(data);

      if (resActionBox) {
        resActionBox.className = `action-recommendation-box ${isDent ? "" : "crack"}`;
        if (resRecommendedAction) resRecommendedAction.textContent = data.recommended_action;
      }
    }

    // Populate Technical Details Accordion
    if (telInspectionId) telInspectionId.textContent = data.inspection_id || "—";
    if (telProcessingTime) telProcessingTime.textContent = data.processing_time_ms ? `${data.processing_time_ms} ms` : "—";
    if (telAircraftStatus) telAircraftStatus.textContent = data.aircraft_image_status || (isSupported ? "SUPPORTED" : "UNSUPPORTED");
    if (telPresenceStatus) telPresenceStatus.textContent = data.damage_presence_status || "NOT_ASSESSED";
    if (telDamageType) telDamageType.textContent = (data.damage_type || "NOT_ASSESSED").toUpperCase();
    if (telUncertaintyStatus) telUncertaintyStatus.textContent = data.uncertainty_status || "—";
    if (telDecisionMargin) {
      telDecisionMargin.textContent = data.margin_from_decision !== null && data.margin_from_decision !== undefined
        ? `\u00B1 ${data.margin_from_decision.toFixed(4)}`
        : "N/A";
    }
    if (telModelVersion) telModelVersion.textContent = data.model_version || "VGG16-AeroDamage-v1.0";
    if (resSafetyDisclaimer) resSafetyDisclaimer.textContent = data.disclaimer;
  }

  function renderScoreBars(data) {
    if (resScoresBlock) resScoresBlock.style.display = "block";

    const dentPct = ((data.raw_dent_score ?? 0) * 100).toFixed(2);
    const crackPct = ((data.raw_crack_score ?? 0) * 100).toFixed(2);

    if (valScoreDent) valScoreDent.textContent = `${dentPct}%`;
    if (barScoreDent) barScoreDent.style.width = `${dentPct}%`;

    if (valScoreCrack) valScoreCrack.textContent = `${crackPct}%`;
    if (barScoreCrack) barScoreCrack.style.width = `${crackPct}%`;
  }

  // =========================================================================
  // Report Generation & Export
  // =========================================================================
  btnExportReport?.addEventListener("click", async () => {
    if (!currentInspectionResult) return;
    try {
      const res = await fetch("/api/inspection/report/html", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentInspectionResult),
      });
      if (!res.ok) throw new Error("Report generation failed");
      const htmlText = await res.text();
      const reportWindow = window.open("", "_blank");
      if (reportWindow) {
        reportWindow.document.open();
        reportWindow.document.write(htmlText);
        reportWindow.document.close();
      } else {
        showAlert("Popup blocked. Please permit popups to view printable report.", "amber");
      }
    } catch (e) {
      showAlert(`Report generation error: ${e.message}`, "crimson");
    }
  });

  btnExportJson?.addEventListener("click", () => {
    if (!currentInspectionResult) return;
    const jsonStr = JSON.stringify(currentInspectionResult, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `inspection_${currentInspectionResult.inspection_id || "record"}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // =========================================================================
  // Session History Management
  // =========================================================================
  function saveInspectionToHistory(data) {
    const record = {
      inspection_id: data.inspection_id,
      timestamp: data.timestamp,
      filename: currentAnalysisFilename,
      predicted_class: (data.predicted_class || "UNSUPPORTED").toUpperCase(),
      model_score: data.model_score !== null && data.model_score !== undefined
        ? `${(data.model_score * 100).toFixed(1)}%`
        : "N/A",
      status: data.status,
      full_data: data,
    };

    inspectionHistory.unshift(record);
    if (inspectionHistory.length > 25) inspectionHistory.pop();

    try {
      sessionStorage.setItem("aerodamage_inspections", JSON.stringify(inspectionHistory));
    } catch (e) {}

    renderHistoryTable();
  }

  function renderHistoryTable() {
    if (!historyTbody) return;

    if (historyCountBadge) {
      historyCountBadge.textContent = `${inspectionHistory.length} Records`;
    }

    if (inspectionHistory.length === 0) {
      historyTbody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
            No inspections performed during this session yet.
          </td>
        </tr>
      `;
      return;
    }

    historyTbody.innerHTML = inspectionHistory.map((item, idx) => {
      const cls = item.predicted_class.toLowerCase();
      let pillClass = "unsupported";
      if (cls === "dent") pillClass = "dent";
      else if (cls === "crack") pillClass = "crack";
      else if (cls === "uncertain") pillClass = "uncertain";

      const timeFormatted = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : "—";

      return `
        <tr>
          <td style="font-weight: 600; color: var(--text-pure);">${item.inspection_id}</td>
          <td>${timeFormatted}</td>
          <td>${item.filename}</td>
          <td><span class="status-pill ${pillClass}">${item.predicted_class}</span></td>
          <td>${item.model_score}</td>
          <td>${item.status}</td>
          <td>
            <button class="btn btn-sm btn-outline btn-history-report" data-idx="${idx}">Report</button>
          </td>
        </tr>
      `;
    }).join("");

    document.querySelectorAll(".btn-history-report").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const idx = parseInt(e.target.dataset.idx, 10);
        const record = inspectionHistory[idx];
        if (record?.full_data) {
          try {
            const res = await fetch("/api/inspection/report/html", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(record.full_data),
            });
            if (!res.ok) throw new Error("Report generation failed");
            const htmlText = await res.text();
            const w = window.open("", "_blank");
            w.document.open();
            w.document.write(htmlText);
            w.document.close();
          } catch (err) {
            showAlert(`Failed to open report: ${err.message}`, "crimson");
          }
        }
      });
    });
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
        showAlert(`Failed to load BLIP benchmark image: ${err.message}`, "crimson");
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
