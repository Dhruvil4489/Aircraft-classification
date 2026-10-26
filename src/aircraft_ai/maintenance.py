"""AeroDamage AI — Maintenance Decision Support and Recommendation Engine.

Provides technically grounded, damage-specific maintenance decision support for
predicted dents and cracks on aircraft structures.

CRITICAL SAFETY DIRECTIVE:
Aircraft structural repair is safety-critical. This engine NEVER:
- Invents allowable dent dimensions, crack length limits, or repair depth limits.
- Invents Structural Repair Manual (SRM) section numbers or manufacturer instructions.
- Authorizes return to service, airworthiness release, or DIY/improvised repairs.
- Treats a visible defect as cosmetic based only on image classification.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aircraft_ai.maintenance")


class AssessmentStatus(str, Enum):
    """Authoritative maintenance disposition and inspection assessment statuses."""

    AWAITING_INSPECTION = "awaiting_inspection"
    MEASUREMENTS_REQUIRED = "measurements_required"
    MAINTENANCE_REFERENCE_REQUIRED = "maintenance_reference_required"
    QUALIFIED_ASSESSMENT_REQUIRED = "qualified_assessment_required"
    AWAITING_AUTHORIZED_DISPOSITION = "awaiting_authorized_disposition"


class SourceStatus(str, Enum):
    """Traceability of recommended action items."""

    GENERAL_INSPECTION_GUIDANCE = "general_inspection_guidance"
    VERIFIED_MAINTENANCE_REFERENCE = "verified_maintenance_reference"
    PRECAUTIONARY_ESCALATION = "precautionary_escalation"
    MANDATORY_REGULATORY_POLICY = "mandatory_regulatory_policy"


@dataclass
class InspectionContext:
    """Inspector-supplied physical and contextual information."""

    aircraft_manufacturer: Optional[str] = None
    aircraft_model: Optional[str] = None
    component_id: Optional[str] = None
    structural_location: Optional[str] = None
    material: Optional[str] = None
    damage_length_mm: Optional[float] = None
    damage_width_mm: Optional[float] = None
    damage_depth_mm: Optional[float] = None
    measurement_method: Optional[str] = None
    maintenance_reference: Optional[str] = None
    inspection_notes: Optional[str] = None
    is_near_joint_or_fastener: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendedActionStep:
    """Single ordered inspection action step."""

    step: int
    action: str
    source_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaintenanceReferenceItem:
    """Traceable maintenance manual or engineering document reference."""

    document_title: str
    document_id: str
    applicable_section: str
    revision: Optional[str] = None
    page: Optional[str] = None
    verified: bool = False
    applicability_notes: str = ""
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DamageRecommendation:
    """Structured damage assessment and maintenance recommendation response."""

    inspection_id: Optional[str]
    predicted_class: Optional[str]
    model_score: Optional[float]
    score_type: str
    uncertainty_status: str
    assessment_status: str
    qualified_heading: str
    summary: str
    recommended_actions: List[RecommendedActionStep]
    maintenance_references: List[MaintenanceReferenceItem]
    missing_information: List[str]
    limitations: List[str]
    rationale: str
    context_used: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "predicted_class": self.predicted_class,
            "model_score": self.model_score,
            "score_type": self.score_type,
            "uncertainty_status": self.uncertainty_status,
            "assessment_status": self.assessment_status,
            "qualified_heading": self.qualified_heading,
            "summary": self.summary,
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "maintenance_references": [r.to_dict() for r in self.maintenance_references],
            "missing_information": self.missing_information,
            "limitations": self.limitations,
            "rationale": self.rationale,
            "context_used": self.context_used,
        }


# Standard authoritative guidance notes
MISSING_REFERENCE_NOTICE = (
    "Applicable approved maintenance data is not available in this system. "
    "Obtain the current approved documentation for the aircraft and affected "
    "component before determining the repair."
)

STANDARD_SAFETY_LIMITATIONS = [
    "Image classification alone cannot determine allowable damage limits, crack length limits, or authorize a repair.",
    "Do not instruct or attempt to fill, sand, hammer, heat, drill, weld, bond, patch, or repair aircraft structure based on AI output.",
    "Physical damage measurements must be taken by qualified personnel using calibrated aerospace inspection tools.",
    "This AI-assisted assessment does not constitute airworthiness authorization or approval for return to service.",
]

# Verified reference database for authorized publications
# Only genuine, non-fabricated references are listed here.
VERIFIED_REFERENCE_CATALOG: Dict[str, Dict[str, Any]] = {
    "FAA_AC_43_13_1B": {
        "document_title": "FAA Advisory Circular AC 43.13-1B: Acceptable Methods, Techniques, and Practices — Aircraft Inspection and Repair",
        "document_id": "FAA-AC-43.13-1B",
        "applicable_section": "Chapter 4: Metal Structure, Welding, and Brazing / Chapter 5: Nondestructive Inspection (NDI)",
        "revision": "Change 1",
        "verified": True,
        "applicability_notes": "General standard aerospace inspection guidance. Does NOT supersede aircraft manufacturer Structural Repair Manuals (SRM).",
    },
    "FAA_AC_43_13_2B": {
        "document_title": "FAA Advisory Circular AC 43.13-2B: Acceptable Methods, Techniques, and Practices — Aircraft Alterations",
        "document_id": "FAA-AC-43.13-2B",
        "applicable_section": "Chapter 1: Structural Data",
        "revision": "Current",
        "verified": True,
        "applicability_notes": "General alteration guidance. Applicable only when approved technical data is referenced.",
    },
    "ATA_SPEC_100_CH51": {
        "document_title": "Air Transport Association (ATA) Specification 100 / iSpec 2200",
        "document_id": "ATA-iSpec-2200-CH51",
        "applicable_section": "Chapter 51: Standard Practices and Structures — General",
        "revision": "Standard Industry Specification",
        "verified": True,
        "applicability_notes": "Standard industry structural classification taxonomy for damage assessment and NDT reporting.",
    },
}

# Critical structural keywords that indicate joint, fastener, frame, or stringer boundaries
CRITICAL_STRUCTURAL_KEYWORDS = [
    "joint",
    "lap",
    "butt",
    "seam",
    "fastener",
    "rivet",
    "bolt",
    "frame",
    "stringer",
    "longerons",
    "rib",
    "spar",
    "bulkhead",
    "doubler",
    "splice",
    "radius",
    "joggle",
]


def _normalize_context(context_input: Optional[Any]) -> InspectionContext:
    """Normalize input into an InspectionContext dataclass instance."""
    if context_input is None:
        return InspectionContext()
    if isinstance(context_input, InspectionContext):
        return context_input
    if isinstance(context_input, dict):
        # Validate measurements: ignore NaNs or negative numbers
        def _clean_float(val: Any) -> Optional[float]:
            if val is None or val == "":
                return None
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f) or f < 0:
                    return None
                return f
            except (ValueError, TypeError):
                return None

        def _clean_str(val: Any) -> Optional[str]:
            if val is None:
                return None
            s = str(val).strip()
            return s if s else None

        def _clean_bool(val: Any) -> Optional[bool]:
            if val is None:
                return None
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)

        return InspectionContext(
            aircraft_manufacturer=_clean_str(context_input.get("aircraft_manufacturer")),
            aircraft_model=_clean_str(context_input.get("aircraft_model")),
            component_id=_clean_str(context_input.get("component_id")),
            structural_location=_clean_str(context_input.get("structural_location")),
            material=_clean_str(context_input.get("material")),
            damage_length_mm=_clean_float(context_input.get("damage_length_mm")),
            damage_width_mm=_clean_float(context_input.get("damage_width_mm")),
            damage_depth_mm=_clean_float(context_input.get("damage_depth_mm")),
            measurement_method=_clean_str(context_input.get("measurement_method")),
            maintenance_reference=_clean_str(context_input.get("maintenance_reference")),
            inspection_notes=_clean_str(context_input.get("inspection_notes")),
            is_near_joint_or_fastener=_clean_bool(context_input.get("is_near_joint_or_fastener")),
        )
    return InspectionContext()


def _is_near_critical_feature(context: InspectionContext) -> bool:
    """Determine if inspection context indicates proximity to structural joints or fasteners."""
    if context.is_near_joint_or_fastener is True:
        return True

    text_to_check = " ".join(
        filter(
            None,
            [
                context.structural_location,
                context.component_id,
                context.inspection_notes,
            ],
        )
    ).lower()

    return any(re.search(rf"\b{kw}\b", text_to_check) for kw in CRITICAL_STRUCTURAL_KEYWORDS)


def _resolve_references(user_ref: Optional[str]) -> List[MaintenanceReferenceItem]:
    """Retrieve verified maintenance reference if applicable, or mark as unavailable."""
    references: List[MaintenanceReferenceItem] = []

    # Always attach the baseline ATA Chapter 51 reference for standard damage classification
    ata_ref = VERIFIED_REFERENCE_CATALOG["ATA_SPEC_100_CH51"]
    references.append(
        MaintenanceReferenceItem(
            document_title=ata_ref["document_title"],
            document_id=ata_ref["document_id"],
            applicable_section=ata_ref["applicable_section"],
            revision=ata_ref["revision"],
            verified=True,
            applicability_notes=ata_ref["applicability_notes"],
        )
    )

    if not user_ref:
        return references

    norm_ref = user_ref.strip().upper()

    # Check for recognized standard advisories
    if "43.13-1B" in norm_ref or "AC 43.13" in norm_ref:
        item = VERIFIED_REFERENCE_CATALOG["FAA_AC_43_13_1B"]
        references.append(
            MaintenanceReferenceItem(
                document_title=item["document_title"],
                document_id=item["document_id"],
                applicable_section=item["applicable_section"],
                revision=item["revision"],
                verified=True,
                applicability_notes=item["applicability_notes"],
            )
        )
    else:
        # User entered a reference, but its existence/applicability cannot be verified in this system
        references.append(
            MaintenanceReferenceItem(
                document_title=f"Inspector-Provided Reference: {user_ref}",
                document_id="UNVERIFIED-INSPECTOR-INPUT",
                applicable_section="Section verification pending engineering review",
                verified=False,
                applicability_notes=MISSING_REFERENCE_NOTICE,
            )
        )

    return references


def _determine_missing_information(
    context: InspectionContext,
    is_dent: bool,
) -> List[str]:
    """Identify missing technical parameters essential for aircraft maintenance disposition."""
    missing: List[str] = []

    if not context.aircraft_manufacturer:
        missing.append("aircraft_manufacturer")
    if not context.aircraft_model:
        missing.append("aircraft_model (variant/type identification)")
    if not context.component_id:
        missing.append("component_id (part number or structural zone)")
    if not context.structural_location:
        missing.append("structural_location (station, frame, stringer, or skin bay)")
    if not context.material:
        missing.append("material_specification (e.g., 2024-T3, 7075-T6, CFRP composite)")

    if is_dent:
        missing_measurements: List[str] = []
        if context.damage_depth_mm is None:
            missing_measurements.append("depth")
        if context.damage_length_mm is None:
            missing_measurements.append("length")
        if context.damage_width_mm is None:
            missing_measurements.append("width")
        if missing_measurements:
            missing.append(f"damage_measurements ({', '.join(missing_measurements)})")
    else:
        # Crack
        if context.damage_length_mm is None:
            missing.append("crack_length_measurement (calibrated physical dimension)")

    if not context.maintenance_reference or "UNVERIFIED" in str(context.maintenance_reference):
        missing.append("applicable_approved_maintenance_reference (current revision manufacturer SRM/AMM)")

    return missing


def evaluate_dent_recommendation(
    model_score: Optional[float],
    uncertainty_status: str,
    context: InspectionContext,
    inspection_id: Optional[str] = None,
) -> DamageRecommendation:
    """Implement the dedicated Dent-specific recommendation policy per prompt Section 3."""
    score_type = "Uncalibrated Sigmoid Activation"
    missing_info = _determine_missing_information(context, is_dent=True)
    references = _resolve_references(context.maintenance_reference)
    is_near_joint = _is_near_critical_feature(context)

    has_depth = context.damage_depth_mm is not None
    has_dimensions = context.damage_length_mm is not None and context.damage_width_mm is not None
    has_aircraft_info = bool(context.aircraft_model and context.component_id)
    has_verified_ref = any(r.verified and "UNVERIFIED" not in r.document_id for r in references if "FAA-AC" in r.document_id or "SRM" in r.document_title)

    # Base steps common to initial assessment
    recommended_actions: List[RecommendedActionStep] = []
    step_num = 1

    # Step 1: Initial assessment
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=(
                "Review original inspection photograph and Grad-CAM localized attention zone. "
                "Record inspection location, viewing angle, and image metadata in the aircraft technical log."
            ),
            source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
        )
    )
    step_num += 1

    # Decision guidance logic (States A, B, C, D)
    if not (has_depth and has_dimensions):
        # State A: Further assessment required — dimensions not established
        assessment_status = AssessmentStatus.MEASUREMENTS_REQUIRED.value
        qualified_heading = "Possible Surface Dent Detected"
        summary = (
            "A possible surface dent was identified by the model. Its dimensions, depth, contour, "
            "and structural significance have not been established from the 2D photograph alone."
        )
        rationale = (
            "Damage is suspected, but physical dimensions (depth, width, length) and exact structural "
            "location are unknown. 2D image classification cannot estimate depth or allowable damage limits."
        )

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Measure dent length, width, and maximum depth using an approved aerospace tool "
                    "(e.g., dial depth gauge with bridging bar, depth micrometer, or optical profilometer)."
                ),
                source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
            )
        )
        step_num += 1

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Examine surrounding skin for secondary damage: check for paint flaking, creases, "
                    "associated cracking at dent edges, or buckling."
                ),
                source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
            )
        )
        step_num += 1

    elif is_near_joint:
        # State C: Qualified structural repair assessment required — critical boundary
        assessment_status = AssessmentStatus.QUALIFIED_ASSESSMENT_REQUIRED.value
        qualified_heading = "Surface Dent Detected — Proximity to Critical Structural Feature"
        summary = (
            f"A surface dent ({context.damage_length_mm:.1f} \u00D7 {context.damage_width_mm:.1f} mm, "
            f"depth {context.damage_depth_mm:.2f} mm) is located near a structural joint, frame, stringer, or fastener line. "
            "Proximity to structural members significantly reduces allowable damage limits due to stress concentration."
        )
        rationale = (
            "Damage is located within or adjacent to a primary structural load path or fastener margin. "
            "Allowable damage limits for open skin bays do not apply near structural joints or stiffeners. "
            "Immediate assessment by qualified structural repair personnel is required."
        )

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Perform detailed inspection of adjacent fasteners, rivets, and frame/stringer attachment points "
                    "for fastener tilting, head shear, rivet hole distortion, or internal joint gap deformation."
                ),
                source_status=SourceStatus.PRECAUTIONARY_ESCALATION.value,
            )
        )
        step_num += 1

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Escalate finding to licensed Aircraft Structural Repair Engineer / Stress Engineering. "
                    "Compare physical measurements against the specific structural joint criteria in the applicable Structural Repair Manual (SRM)."
                ),
                source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
            )
        )
        step_num += 1

    elif not has_aircraft_info or not has_verified_ref:
        # State B: Maintenance documentation review required
        assessment_status = AssessmentStatus.MAINTENANCE_REFERENCE_REQUIRED.value
        qualified_heading = "Surface Dent Measured — Maintenance Documentation Review Required"
        summary = (
            f"Dent dimensions recorded ({context.damage_length_mm:.1f} \u00D7 {context.damage_width_mm:.1f} mm, "
            f"depth {context.damage_depth_mm:.2f} mm), but applicable approved allowable-damage criteria "
            "for this specific aircraft model and structural panel have not been verified in the system."
        )
        rationale = (
            "Physical dimensions have been recorded, but allowable damage criteria cannot be established without "
            "the authoritative aircraft Structural Repair Manual (SRM Chapter 51 / 53) for the specific variant and skin panel."
        )

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Obtain the current approved Structural Repair Manual (SRM) or Aircraft Maintenance Manual (AMM) "
                    "for the exact aircraft type, variant, and component location."
                ),
                source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
            )
        )
        step_num += 1

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Have a qualified Aircraft Maintenance Engineer (AME / Part 66 / A&P) compare measured length, "
                    "width, depth, and edge distance against the manufacturer's allowable damage limit (ADL) charts."
                ),
                source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
            )
        )
        step_num += 1

    else:
        # State D: Documentation-supported disposition available
        assessment_status = AssessmentStatus.AWAITING_AUTHORIZED_DISPOSITION.value
        qualified_heading = "Surface Dent Documented — Awaiting Authorized Maintenance Disposition"
        summary = (
            f"Complete inspection parameters and verified maintenance reference documented for {context.aircraft_model} "
            f"({context.component_id}). Measured dimensions: {context.damage_length_mm:.1f} \u00D7 {context.damage_width_mm:.1f} mm, "
            f"depth {context.damage_depth_mm:.2f} mm. Qualified maintenance personnel must evaluate allowable criteria."
        )
        rationale = (
            "Inspection parameters and maintenance references are documented. Qualified maintenance personnel "
            "can evaluate documented criteria. The software does NOT independently authorize return to service."
        )

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "Qualified structural repair personnel must compare measured depth and dimensions directly against "
                    f"the allowable damage limits in the verified maintenance documentation ({context.maintenance_reference})."
                ),
                source_status=SourceStatus.VERIFIED_MAINTENANCE_REFERENCE.value,
            )
        )
        step_num += 1

        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "If damage is within allowable limits, record defect disposition in the aircraft technical logbook "
                    "with required repetitive inspection intervals. If damage exceeds limits, prepare an approved engineering repair disposition."
                ),
                source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
            )
        )
        step_num += 1

    # Final universal qualification step
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=(
                "Obtain official sign-off from authorized maintenance personnel prior to closing the inspection record. "
                "Do not use AI output as authorization to return aircraft to service."
            ),
            source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
        )
    )

    return DamageRecommendation(
        inspection_id=inspection_id,
        predicted_class="dent",
        model_score=model_score,
        score_type=score_type,
        uncertainty_status=uncertainty_status,
        assessment_status=assessment_status,
        qualified_heading=qualified_heading,
        summary=summary,
        recommended_actions=recommended_actions,
        maintenance_references=references,
        missing_information=missing_info,
        limitations=list(STANDARD_SAFETY_LIMITATIONS),
        rationale=rationale,
        context_used=context.to_dict(),
    )


def evaluate_crack_recommendation(
    model_score: Optional[float],
    uncertainty_status: str,
    context: InspectionContext,
    inspection_id: Optional[str] = None,
) -> DamageRecommendation:
    """Implement the dedicated Crack-specific recommendation policy per prompt Section 4.

    A suspected structural crack must NEVER be treated as a harmless surface defect.
    """
    score_type = "Uncalibrated Sigmoid Activation"
    missing_info = _determine_missing_information(context, is_dent=False)
    references = _resolve_references(context.maintenance_reference)
    is_near_joint = _is_near_critical_feature(context)

    # For structural cracks, safety requires immediate escalation to qualified personnel
    has_verified_ref = any(r.verified and "UNVERIFIED" not in r.document_id for r in references if "FAA-AC" in r.document_id or "SRM" in r.document_title)

    if has_verified_ref and context.aircraft_model and context.component_id:
        assessment_status = AssessmentStatus.AWAITING_AUTHORIZED_DISPOSITION.value
    else:
        assessment_status = AssessmentStatus.QUALIFIED_ASSESSMENT_REQUIRED.value

    qualified_heading = "Suspected Crack Detected — Maintenance Assessment Required"
    summary = (
        "A suspected crack discontinuity was detected. Structural cracks represent severe stress "
        "concentrations and fatigue propagation hazards. Immediate verification and non-destructive "
        "testing (NDT) by qualified maintenance personnel is mandatory."
    )
    rationale = (
        "Cracks in aircraft primary or secondary structures carry high airworthiness risks. "
        "Visual appearance or model confidence cannot establish crack depth, tip arrest, material yield, "
        "or propagation rate. Operational restrictions and approved procedures must be evaluated immediately."
    )

    recommended_actions: List[RecommendedActionStep] = []
    step_num = 1

    # Step 1: Immediate quarantine and notice
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=(
                "Flag component as SUSPECTED CRACK — MAINTENANCE ASSESSMENT REQUIRED. Notify lead inspector "
                "and follow approved aircraft maintenance procedures regarding operational status."
            ),
            source_status=SourceStatus.PRECAUTIONARY_ESCALATION.value,
        )
    )
    step_num += 1

    # Step 2: Confirm genuine crack vs artifact
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=(
                "Have qualified personnel physically clean the area and verify whether the indication is a genuine "
                "material crack, a surface scratch, a paint/coating fracture, or a sealant boundary."
            ),
            source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
        )
    )
    step_num += 1

    # Step 3: Material and geometry specific NDT selection
    ndt_method_note = (
        "Select suitable inspection or NDT methods according to the applicable approved maintenance procedure "
        "(e.g., High-Frequency Eddy Current [HFEC] for surface cracks in conductive aluminum alloys, "
        "Ultrasonic Testing [UT] for subsurface propagation or composite delamination, or Dye Penetrant [PT] where permitted)."
    )
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=ndt_method_note,
            source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
        )
    )
    step_num += 1

    # Step 4: Proximity to joints/fasteners
    if is_near_joint:
        recommended_actions.append(
            RecommendedActionStep(
                step=step_num,
                action=(
                    "CRITICAL: The crack indication is near a structural joint, fastener hole, or stiffener boundary. "
                    "Check for fastener hole cracking, multi-site damage (MSD), and fastener head integrity per SRM Chapter 51 / 53."
                ),
                source_status=SourceStatus.PRECAUTIONARY_ESCALATION.value,
            )
        )
        step_num += 1

    # Step 5: Engineering repair disposition
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=(
                "Do NOT perform generic stop-drilling, grinding, metal patching, welding, or bonding without authorized engineering disposition. "
                "A formal approved repair procedure or OEM engineering disposition must be obtained."
            ),
            source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
        )
    )
    step_num += 1

    # Step 6: Authorization requirement
    recommended_actions.append(
        RecommendedActionStep(
            step=step_num,
            action=(
                "Airworthiness determination, grounding, or flight clearance must be executed solely by licensed "
                "maintenance personnel in accordance with approved airline/OEM operating manuals."
            ),
            source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
        )
    )

    return DamageRecommendation(
        inspection_id=inspection_id,
        predicted_class="crack",
        model_score=model_score,
        score_type=score_type,
        uncertainty_status=uncertainty_status,
        assessment_status=assessment_status,
        qualified_heading=qualified_heading,
        summary=summary,
        recommended_actions=recommended_actions,
        maintenance_references=references,
        missing_information=missing_info,
        limitations=list(STANDARD_SAFETY_LIMITATIONS),
        rationale=rationale,
        context_used=context.to_dict(),
    )


def evaluate_uncertain_recommendation(
    model_score: Optional[float],
    uncertainty_status: str,
    context: InspectionContext,
    inspection_id: Optional[str] = None,
    uncertainty_reason: Optional[str] = None,
) -> DamageRecommendation:
    """Implement recommendation policy for uncertain/ambiguous model predictions."""
    score_type = "Uncalibrated Sigmoid Activation"
    missing_info = _determine_missing_information(context, is_dent=True)
    references = _resolve_references(context.maintenance_reference)

    qualified_heading = "Uncertain Visual Evidence — Manual Inspection Required"
    summary = (
        "Model prediction is within the uncertainty margin near the decision threshold. "
        "The visual features do not provide clear evidence to distinguish whether the surface indication "
        "is a dent, crack, scratch, or benign contour variation."
    )
    rationale = (
        f"Classification confidence is ambiguous ({uncertainty_reason or 'activation near decision threshold'}). "
        "Automated decision-making is withheld. Hands-on physical inspection and optical magnification are required."
    )

    recommended_actions = [
        RecommendedActionStep(
            step=1,
            action=(
                "Do NOT rely on automated classification. Conduct General Visual Inspection (GVI) and Detailed Inspection (DET) "
                "under adequate lighting and 10\u00D7 optical magnification."
            ),
            source_status=SourceStatus.PRECAUTIONARY_ESCALATION.value,
        ),
        RecommendedActionStep(
            step=2,
            action=(
                "Clean the panel surface to remove dirt, grease, fluid residue, or loose sealant that may obscure surface features."
            ),
            source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
        ),
        RecommendedActionStep(
            step=3,
            action=(
                "Perform tactile inspection with a clean fingertip or calibrated straightedge to detect physical depth or surface depression."
            ),
            source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
        ),
        RecommendedActionStep(
            step=4,
            action=(
                "If a linear defect is suspected, apply an approved non-destructive testing (NDT) method per the applicable maintenance manual."
            ),
            source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
        ),
    ]

    return DamageRecommendation(
        inspection_id=inspection_id,
        predicted_class="uncertain",
        model_score=model_score,
        score_type=score_type,
        uncertainty_status=uncertainty_status,
        assessment_status=AssessmentStatus.QUALIFIED_ASSESSMENT_REQUIRED.value,
        qualified_heading=qualified_heading,
        summary=summary,
        recommended_actions=recommended_actions,
        maintenance_references=references,
        missing_information=missing_info,
        limitations=list(STANDARD_SAFETY_LIMITATIONS),
        rationale=rationale,
        context_used=context.to_dict(),
    )


def evaluate_unsupported_or_ood_recommendation(
    predicted_class: Optional[str],
    model_score: Optional[float],
    uncertainty_status: str,
    context: InspectionContext,
    inspection_id: Optional[str] = None,
    rejection_reason: Optional[str] = None,
) -> DamageRecommendation:
    """Implement recommendation policy for unsupported defect classes or out-of-domain images."""
    score_type = "Domain Screening / Class Support Validation"
    missing_info = _determine_missing_information(context, is_dent=True)
    references = _resolve_references(context.maintenance_reference)

    if predicted_class is not None and predicted_class.lower() not in ("dent", "crack", "uncertain"):
        qualified_heading = f"Unsupported Defect Class: '{predicted_class.upper()}'"
        summary = (
            f"The requested defect class '{predicted_class}' is not supported by the validated VGG16 model, "
            "which only classifies DENT and CRACK. The system will not invent predictions for unsupported damage modes."
        )
        rationale = (
            f"Defect type '{predicted_class}' requires specialized sensor data or an appropriately trained and validated model. "
            "Manual assessment by qualified aircraft maintenance personnel using the Structural Repair Manual is required."
        )
    else:
        qualified_heading = "Image Screening Rejection — Aircraft Domain Not Verified"
        summary = (
            rejection_reason
            or "The uploaded image could not be verified as a valid aircraft structural surface photograph. "
            "Damage classification and repair recommendations are withheld."
        )
        rationale = (
            "Screening model detected non-aircraft imagery or unverified damage presence. "
            "Binary classification is bypassed to avoid spurious predictions."
        )

    recommended_actions = [
        RecommendedActionStep(
            step=1,
            action=(
                "Upload a high-resolution, properly illuminated technical photograph of the aircraft structural panel under review."
            ),
            source_status=SourceStatus.GENERAL_INSPECTION_GUIDANCE.value,
        ),
        RecommendedActionStep(
            step=2,
            action=(
                "For damage types other than dents or cracks (e.g., corrosion, delamination, lightning strike, fastener shear), "
                "refer directly to the applicable Structural Repair Manual (SRM Chapter 51) and schedule hands-on NDT."
            ),
            source_status=SourceStatus.MANDATORY_REGULATORY_POLICY.value,
        ),
    ]

    return DamageRecommendation(
        inspection_id=inspection_id,
        predicted_class=predicted_class,
        model_score=model_score,
        score_type=score_type,
        uncertainty_status=uncertainty_status,
        assessment_status=AssessmentStatus.AWAITING_INSPECTION.value,
        qualified_heading=qualified_heading,
        summary=summary,
        recommended_actions=recommended_actions,
        maintenance_references=references,
        missing_information=missing_info,
        limitations=list(STANDARD_SAFETY_LIMITATIONS),
        rationale=rationale,
        context_used=context.to_dict(),
    )


def get_damage_recommendation(
    predicted_class: Optional[str],
    model_score: Optional[float] = None,
    uncertainty_status: str = "confident",
    context: Optional[Any] = None,
    inspection_id: Optional[str] = None,
    uncertainty_reason: Optional[str] = None,
) -> DamageRecommendation:
    """Main entrypoint for the maintenance decision support system.

    Parameters
    ----------
    predicted_class : Optional[str]
        Defect class predicted by the model ("dent", "crack", "uncertain", or None).
    model_score : Optional[float]
        Predicted model score (e.g., probability).
    uncertainty_status : str
        Uncertainty flag ("confident", "uncertain", "out_of_domain", "damage_presence_unverified").
    context : Optional[Any]
        InspectionContext dataclass or dict containing user-supplied inspector parameters.
    inspection_id : Optional[str]
        Unique inspection identifier.
    uncertainty_reason : Optional[str]
        Optional explanation for uncertain or rejected model state.

    Returns
    -------
    DamageRecommendation
        Structured maintenance recommendation with ordered actions, status, references, and limitations.
    """
    ctx = _normalize_context(context)

    # 1. Check for OOD or unverified image status
    if uncertainty_status in ("out_of_domain", "damage_presence_unverified") or predicted_class is None:
        return evaluate_unsupported_or_ood_recommendation(
            predicted_class=predicted_class,
            model_score=model_score,
            uncertainty_status=uncertainty_status,
            context=ctx,
            inspection_id=inspection_id,
            rejection_reason=uncertainty_reason,
        )

    norm_class = str(predicted_class).strip().lower()

    # 2. Check for unsupported classes
    if norm_class not in ("dent", "crack", "uncertain"):
        logger.warning("Unsupported defect class requested: %s", norm_class)
        return evaluate_unsupported_or_ood_recommendation(
            predicted_class=norm_class,
            model_score=model_score,
            uncertainty_status="unsupported_class",
            context=ctx,
            inspection_id=inspection_id,
            rejection_reason=f"Defect class '{norm_class}' is not supported by the model.",
        )

    # 3. Check for uncertain prediction
    if norm_class == "uncertain" or uncertainty_status == "uncertain":
        return evaluate_uncertain_recommendation(
            model_score=model_score,
            uncertainty_status=uncertainty_status,
            context=ctx,
            inspection_id=inspection_id,
            uncertainty_reason=uncertainty_reason,
        )

    # 4. Dispatch to damage-specific policy
    if norm_class == "dent":
        return evaluate_dent_recommendation(
            model_score=model_score,
            uncertainty_status=uncertainty_status,
            context=ctx,
            inspection_id=inspection_id,
        )
    elif norm_class == "crack":
        return evaluate_crack_recommendation(
            model_score=model_score,
            uncertainty_status=uncertainty_status,
            context=ctx,
            inspection_id=inspection_id,
        )

    # Fallback safe handling
    return evaluate_uncertain_recommendation(
        model_score=model_score,
        uncertainty_status="uncertain",
        context=ctx,
        inspection_id=inspection_id,
        uncertainty_reason="Unhandled state in recommendation policy.",
    )
