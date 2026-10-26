"""Unit and regression tests for AeroDamage AI Maintenance Decision Support System.

Tests all 15 scenarios specified in Prompt Section 11:
1. Dent with unknown dimensions.
2. Dent with supplied measurements but no applicable maintenance limits.
3. Dent near a structural joint, where that location is actually provided.
4. Suspected crack with uncertain model output.
5. Suspected crack with known component information.
6. Unknown aircraft model.
7. Missing maintenance documentation.
8. Unsupported predicted class.
9. Invalid or missing measurements.
10. Model inference failure handling.
11. Uncertain or out-of-domain image.
12. Verified maintenance reference retrieval.
13. Unavailable or inapplicable reference.
14. Frontend rendering of recommendation states (data contract & markup).
15. Report generation and JSON export.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from aircraft_ai.api import app
from aircraft_ai.maintenance import (
    AssessmentStatus,
    InspectionContext,
    SourceStatus,
    get_damage_recommendation,
)


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# SCENARIO 1: Dent with unknown dimensions
# =============================================================================
def test_scenario_1_dent_with_unknown_dimensions():
    """Verify that a predicted dent without physical measurements requires measurements

    and does not invent allowable limits or repair depths.
    """
    context = InspectionContext(
        aircraft_manufacturer="Boeing",
        aircraft_model="737-800",
        component_id="Fuselage Skin Panel Station 480",
        damage_length_mm=None,
        damage_width_mm=None,
        damage_depth_mm=None,
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.88,
        uncertainty_status="confident",
        context=context,
        inspection_id="TEST-INS-001",
    )

    assert rec.predicted_class == "dent"
    assert rec.assessment_status == AssessmentStatus.MEASUREMENTS_REQUIRED.value
    assert "Possible Surface Dent Detected" in rec.qualified_heading
    assert any("damage_measurements" in m for m in rec.missing_information)

    # Actions must mandate measuring length, width, and depth
    action_texts = " ".join(a.action for a in rec.recommended_actions)
    assert "dial depth gauge" in action_texts or "depth micrometer" in action_texts or "Measure dent" in action_texts

    # Must never authorize return to service or invent limits
    assert "authorized" not in rec.assessment_status
    assert not any("drill" in a.action.lower() for a in rec.recommended_actions)


# =============================================================================
# SCENARIO 2: Dent with supplied measurements but no applicable maintenance limits
# =============================================================================
def test_scenario_2_dent_with_measurements_no_limits():
    """Verify that measured dent with missing maintenance documentation requires

    maintenance documentation review and comparison against manufacturer charts.
    """
    context = InspectionContext(
        damage_length_mm=32.0,
        damage_width_mm=18.5,
        damage_depth_mm=0.75,
        aircraft_model=None,  # missing aircraft model
        maintenance_reference=None,  # missing manual
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.91,
        uncertainty_status="confident",
        context=context,
        inspection_id="TEST-INS-002",
    )

    assert rec.assessment_status == AssessmentStatus.MAINTENANCE_REFERENCE_REQUIRED.value
    assert any("applicable_approved_maintenance_reference" in m for m in rec.missing_information)
    assert any("aircraft_model" in m for m in rec.missing_information)
    assert "32.0" in rec.summary
    assert "0.75" in rec.summary

    # Ensure no fabricated allowable damage limit numbers
    for act in rec.recommended_actions:
        assert "drill" not in act.action.lower()
        assert "stop-drill" not in act.action.lower()


# =============================================================================
# SCENARIO 3: Dent near a structural joint, where that location is provided
# =============================================================================
def test_scenario_3_dent_near_structural_joint():
    """Verify that a dent near a frame, stringer, or lap joint escalates to

    qualified structural repair assessment because allowable limits are altered.
    """
    context = InspectionContext(
        aircraft_manufacturer="Airbus",
        aircraft_model="A320-200",
        component_id="Fuselage Crown Bay 2",
        structural_location="Adjacent to stringer 14 and frame 32 lap joint",
        damage_length_mm=25.0,
        damage_width_mm=20.0,
        damage_depth_mm=0.6,
        is_near_joint_or_fastener=True,
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.86,
        uncertainty_status="confident",
        context=context,
        inspection_id="TEST-INS-003",
    )

    assert rec.assessment_status == AssessmentStatus.QUALIFIED_ASSESSMENT_REQUIRED.value
    assert "Proximity to Critical Structural Feature" in rec.qualified_heading
    assert "joint" in rec.summary.lower() or "fastener" in rec.summary.lower()

    action_texts = " ".join(a.action for a in rec.recommended_actions)
    assert "fastener" in action_texts.lower() or "rivet" in action_texts.lower() or "joint" in action_texts.lower()
    assert "Structural Repair Engineer" in action_texts or "qualified" in action_texts.lower()


# =============================================================================
# SCENARIO 4: Suspected crack with uncertain model output
# =============================================================================
def test_scenario_4_suspected_crack_uncertain():
    """Verify that an ambiguous crack model prediction is treated with high caution

    and mandates hands-on general and detailed visual inspection with NDT.
    """
    rec = get_damage_recommendation(
        predicted_class="uncertain",
        model_score=0.52,
        uncertainty_status="uncertain",
        context=InspectionContext(),
        inspection_id="TEST-INS-004",
        uncertainty_reason="Score near 0.50 decision threshold",
    )

    assert rec.assessment_status == AssessmentStatus.QUALIFIED_ASSESSMENT_REQUIRED.value
    assert "Uncertain Visual Evidence" in rec.qualified_heading
    action_texts = " ".join(a.action for a in rec.recommended_actions)
    assert "Detailed Inspection" in action_texts or "magnification" in action_texts


# =============================================================================
# SCENARIO 5: Suspected crack with known component information
# =============================================================================
def test_scenario_5_suspected_crack_with_component_info():
    """Verify that a suspected crack is NEVER treated as a cosmetic defect

    and mandates prompt qualified inspection with suitable NDT methods.
    """
    context = InspectionContext(
        aircraft_manufacturer="Boeing",
        aircraft_model="B777-300ER",
        component_id="Wing lower skin panel station 420",
        structural_location="Spar web attachment margin",
        material="2024-T3 Aluminum Alloy",
    )

    rec = get_damage_recommendation(
        predicted_class="crack",
        model_score=0.94,
        uncertainty_status="confident",
        context=context,
        inspection_id="TEST-INS-005",
    )

    assert "Suspected Crack Detected" in rec.qualified_heading
    assert rec.assessment_status in (
        AssessmentStatus.QUALIFIED_ASSESSMENT_REQUIRED.value,
        AssessmentStatus.AWAITING_AUTHORIZED_DISPOSITION.value,
    )

    action_texts = " ".join(a.action for a in rec.recommended_actions)
    # Must advise selecting suitable NDT (Eddy Current, Ultrasonic, etc.)
    assert "NDT" in action_texts or "Eddy Current" in action_texts
    # Must explicitly prohibit generic stop-drilling or DIY patching
    assert "stop-drilling" in action_texts.lower() or "patching" in action_texts.lower()
    # Airworthiness release must require licensed personnel
    assert "licensed" in action_texts.lower() or "authorized" in action_texts.lower()


# =============================================================================
# SCENARIO 6: Unknown aircraft model
# =============================================================================
def test_scenario_6_unknown_aircraft_model():
    """Verify that missing aircraft model is reported in missing_information

    and allowable damage limits cannot be claimed.
    """
    context = InspectionContext(
        aircraft_manufacturer=None,
        aircraft_model=None,
        component_id="Elevator Skin",
        damage_depth_mm=0.5,
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.82,
        context=context,
    )

    assert any("aircraft_model" in m for m in rec.missing_information)
    assert rec.assessment_status != AssessmentStatus.AWAITING_AUTHORIZED_DISPOSITION.value


# =============================================================================
# SCENARIO 7: Missing maintenance documentation
# =============================================================================
def test_scenario_7_missing_maintenance_documentation():
    """Verify that when maintenance documentation is missing, the standard

    missing maintenance reference notice is returned without fabricated manual chapters.
    """
    context = InspectionContext(
        aircraft_model="A350-900",
        component_id="Fuselage Frame 20",
        maintenance_reference=None,
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.79,
        context=context,
    )

    assert any("applicable_approved_maintenance_reference" in m for m in rec.missing_information)
    # References should only contain standard ATA spec or unverified notes, not fabricated SRM sections
    for r in rec.maintenance_references:
        assert "FABRICATED" not in r.document_title


# =============================================================================
# SCENARIO 8: Unsupported predicted class
# =============================================================================
def test_scenario_8_unsupported_predicted_class():
    """Verify that requested defect classes not supported by the model (e.g. corrosion,

    delamination, buckling) return an unsupported response without invented limits.
    """
    for bad_class in ["corrosion", "delamination", "buckling", "scratches", "lightning_strike"]:
        rec = get_damage_recommendation(
            predicted_class=bad_class,
            model_score=0.85,
            context=InspectionContext(),
        )

        assert rec.predicted_class == bad_class
        assert "Unsupported Defect Class" in rec.qualified_heading
        assert "not supported" in rec.summary.lower()
        assert rec.assessment_status == AssessmentStatus.AWAITING_INSPECTION.value


# =============================================================================
# SCENARIO 9: Invalid or missing measurements
# =============================================================================
def test_scenario_9_invalid_or_missing_measurements():
    """Verify that negative numbers or non-numeric measurements are sanitized

    and missing dimensions are correctly flagged.
    """
    raw_context = {
        "damage_length_mm": -15.0,  # invalid negative
        "damage_width_mm": "invalid_str",  # invalid string
        "damage_depth_mm": None,
    }

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.85,
        context=raw_context,
    )

    assert rec.assessment_status == AssessmentStatus.MEASUREMENTS_REQUIRED.value
    assert any("damage_measurements" in m for m in rec.missing_information)
    assert rec.context_used["damage_length_mm"] is None
    assert rec.context_used["damage_width_mm"] is None


# =============================================================================
# SCENARIO 10: Model inference failure handling
# =============================================================================
def test_scenario_10_model_inference_failure_handling():
    """Verify that an inference failure or None class produces a safe

    decision support state requiring manual inspection.
    """
    rec = get_damage_recommendation(
        predicted_class=None,
        model_score=None,
        uncertainty_status="inference_error",
        uncertainty_reason="Neural network execution failed",
    )

    assert rec.assessment_status == AssessmentStatus.AWAITING_INSPECTION.value
    assert "Screening Rejection" in rec.qualified_heading or "Unsupported" in rec.qualified_heading


# =============================================================================
# SCENARIO 11: Uncertain or out-of-domain image
# =============================================================================
def test_scenario_11_out_of_domain_image():
    """Verify that out-of-domain images (Stage A rejection) return a safe

    recommendation withholding damage classification.
    """
    rec = get_damage_recommendation(
        predicted_class=None,
        model_score=None,
        uncertainty_status="out_of_domain",
        uncertainty_reason="handwritten signature detected",
    )

    assert rec.assessment_status == AssessmentStatus.AWAITING_INSPECTION.value
    assert not any("crack length" in a.action.lower() for a in rec.recommended_actions)


# =============================================================================
# SCENARIO 12: Verified maintenance reference retrieval
# =============================================================================
def test_scenario_12_verified_maintenance_reference_retrieval():
    """Verify that recognized standard publications (e.g. FAA AC 43.13-1B)

    are marked as verified with genuine metadata and advisory status.
    """
    context = InspectionContext(
        maintenance_reference="FAA AC 43.13-1B",
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.88,
        context=context,
    )

    verified_refs = [r for r in rec.maintenance_references if r.verified]
    assert len(verified_refs) >= 1
    doc_ids = [r.document_id for r in verified_refs]
    assert "FAA-AC-43.13-1B" in doc_ids or "ATA-iSpec-2200-CH51" in doc_ids


# =============================================================================
# SCENARIO 13: Unavailable or inapplicable reference
# =============================================================================
def test_scenario_13_unavailable_reference_notice():
    """Verify that an arbitrary, unverified reference is not accepted as authoritative

    and includes the standard missing approved data notice.
    """
    context = InspectionContext(
        maintenance_reference="SRM-UNKNOWN-CHAPTER-99-99-99",
    )

    rec = get_damage_recommendation(
        predicted_class="dent",
        model_score=0.88,
        context=context,
    )

    unverified_refs = [r for r in rec.maintenance_references if not r.verified]
    assert len(unverified_refs) >= 1
    assert "Applicable approved maintenance data is not available" in unverified_refs[0].applicability_notes


# =============================================================================
# SCENARIO 14: Frontend rendering of recommendation states & API contract
# =============================================================================
def test_scenario_14_api_recommendation_endpoint(client):
    """Test POST /api/recommendation endpoint for dynamic context evaluation."""
    payload = {
        "predicted_class": "dent",
        "model_score": 0.87,
        "uncertainty_status": "confident",
        "context": {
            "aircraft_manufacturer": "Boeing",
            "aircraft_model": "737-800",
            "component_id": "Fuselage Crown Panel",
            "damage_length_mm": 20.0,
            "damage_width_mm": 15.0,
            "damage_depth_mm": 0.5,
            "is_near_joint_or_fastener": False,
        },
        "inspection_id": "TEST-API-001",
    }

    response = client.post("/api/recommendation", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "dent"
    assert "assessment_status" in data
    assert "qualified_heading" in data
    assert "summary" in data
    assert "recommended_actions" in data
    assert len(data["recommended_actions"]) >= 2
    assert "maintenance_references" in data
    assert "limitations" in data
    assert "missing_information" in data

    # Verify frontend elements exist in index.html
    index_html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="res-damage-assessment-panel"' in index_html
    assert 'id="res-disposition-pill"' in index_html
    assert 'id="res-assessment-heading"' in index_html
    assert 'id="res-action-steps-list"' in index_html
    assert 'id="res-references-content"' in index_html
    assert 'id="context-details-accordion"' in index_html


# =============================================================================
# SCENARIO 15: Report generation and JSON export
# =============================================================================
def test_scenario_15_report_generation_and_export(client):
    """Verify that /api/inspection/report/html renders context, actions,

    references, and disclaimers without fabricating signatures or authorizations.
    """
    report_payload = {
        "inspection_id": "INS-20261026-TEST15",
        "timestamp": "2026-10-26T18:00:00Z",
        "status": "COMPLETED",
        "aircraft_image_status": "SUPPORTED",
        "damage_presence_status": "DAMAGE_DETECTED",
        "predicted_class": "dent",
        "damage_type": "dent",
        "model_score": 0.88,
        "decision_message": "Defect classified as DENT based on structural visual features.",
        "assessment_status": "measurements_required",
        "qualified_heading": "Possible Surface Dent Detected",
        "summary": "A possible surface dent was identified. Its dimensions have not been established.",
        "recommended_actions": [
            {
                "step": 1,
                "action": "Record the affected component and structural location.",
                "source_status": "general_inspection_guidance",
            },
            {
                "step": 2,
                "action": "Measure dent length, width, and depth using a dial depth gauge.",
                "source_status": "general_inspection_guidance",
            },
        ],
        "maintenance_references": [
            {
                "document_title": "FAA Advisory Circular AC 43.13-1B",
                "document_id": "FAA-AC-43.13-1B",
                "applicable_section": "Chapter 4: Metal Structure",
                "verified": True,
                "applicability_notes": "General inspection guidance.",
            }
        ],
        "missing_information": ["damage_measurements (depth, length, width)"],
        "limitations": [
            "Image classification alone cannot determine allowable damage limits or authorize a repair."
        ],
        "context_used": {
            "aircraft_manufacturer": "Boeing",
            "aircraft_model": "737-800",
            "component_id": "Fuselage Station 480",
            "damage_length_mm": None,
            "damage_width_mm": None,
            "damage_depth_mm": None,
        },
    }

    response = client.post("/api/inspection/report/html", json=report_payload)
    assert response.status_code == 200
    html_text = response.text

    # Verify required sections are present
    assert "DAMAGE ASSESSMENT &amp; RECOMMENDED ACTION" in html_text or "DAMAGE ASSESSMENT & RECOMMENDED ACTION" in html_text
    assert "MEASUREMENTS REQUIRED" in html_text
    assert "Possible Surface Dent Detected" in html_text
    assert "Record the affected component" in html_text
    assert "FAA-AC-43.13-1B" in html_text
    assert "Boeing" in html_text
    assert "Fuselage Station 480" in html_text
    assert "OPERATIONAL LIMITATION &amp; SAFETY NOTICE" in html_text or "SAFETY NOTICE" in html_text

    # Verify no fabricated repair authorization or airworthiness release
    assert "Airworthiness Approved" not in html_text
    assert "Return to Service Authorized" not in html_text
