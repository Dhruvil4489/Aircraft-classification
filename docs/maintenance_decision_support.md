# AeroDamage AI — Maintenance Decision Support & Damage-Specific Guidance

## 1. System Overview

AeroDamage AI has been extended from a binary defect classifier into an evidence-based, safety-critical aerospace maintenance decision support platform. Following an image inspection, the workstation provides technically grounded, damage-specific maintenance guidance for detected dents and fatigue cracks without making unauthorized airworthiness claims.

---

## 2. Safety Directive & Non-Negotiable Engineering Constraints

Aircraft structural repair is safety-critical. This recommendation engine strictly complies with civil aviation regulations (14 CFR Part 43 / EASA Part-M):

1. **No Invented Limits**: The system never invents allowable dent dimensions, maximum crack lengths, repair depths, SRM section numbers, fastener pitch, or patch geometries.
2. **No Cosmetic Assumptions**: Visible defects are never assumed to be cosmetic simply because they appear small or the model confidence is low.
3. **No Improvised Repairs**: The platform explicitly forbids instructing technicians to drill, stop-drill, sand, hammer, heat, weld, or bond aircraft structure based on 2D image classification.
4. **No Airworthiness Authorization**: Automated outputs serve exclusively as decision support; only licensed Aircraft Maintenance Engineers (AME / A&P / Part 66) can authorize return to service.
5. **Clear Unknown States**: When component geometry, structural location, material, or allowable limits are unknown, the system explicitly states that repair methods cannot yet be established.

---

## 3. Damage-Specific Recommendation Engines

### 3.1 Dent-Specific Recommendation Engine (`evaluate_dent_recommendation`)

A surface dent requires assessment based on location, depth, contour, proximity to structural boundaries, and applicable manufacturer limits.

#### Structured Outcome States:
- **State A: `measurements_required`**
  - *Trigger*: Physical dimensions (length, width, or depth) have not been supplied by an inspector.
  - *Summary*: Explains that 2D image classification cannot determine physical depth or contour.
  - *Next Steps*: Instructs measuring dent length, width, and maximum depth using a dial depth gauge with bridging bar or optical profilometer; checks for secondary buckling or creasing.
- **State B: `maintenance_reference_required`**
  - *Trigger*: Calibrated dimensions are provided, but aircraft model, component ID, or approved maintenance references are missing.
  - *Summary*: Dimensions are recorded, but allowable damage criteria must be compared against the manufacturer's Structural Repair Manual (SRM Chapter 51 / 53).
- **State C: `qualified_assessment_required`**
  - *Trigger*: The dent is located in or adjacent to a primary structural load path (e.g., fastener line, lap joint, frame, stringer, or rib margin).
  - *Summary*: Proximity to structural members alters allowable damage criteria due to stress concentration. Mandates fastener tilt inspection and escalation to structural repair engineering.
- **State D: `awaiting_authorized_disposition`**
  - *Trigger*: Full inspection context, calibrated measurements, and verified maintenance references are available.
  - *Summary*: Qualified maintenance personnel must evaluate documented criteria against the verified reference. Note: software does not authorize return to service.

---

### 3.2 Crack-Specific Recommendation Engine (`evaluate_crack_recommendation`)

A suspected structural crack is **never** treated as a harmless surface defect.

#### Workflow & Escalation:
1. **Immediate Notice**: Flagged as `SUSPECTED CRACK — MAINTENANCE ASSESSMENT REQUIRED`.
2. **Quarantine & Review**: Advise prompt review by qualified personnel following applicable maintenance procedures.
3. **Artifact Disambiguation**: Physical cleaning and 10x optical magnification to differentiate genuine cracks from scratches, paint fractures, or sealant boundaries.
4. **Procedure-Grounded NDT Selection**: Selects suitable NDT methods (e.g., High-Frequency Eddy Current [HFEC] for surface cracks in conductive aluminum alloys, Ultrasonic Testing [UT] for subsurface propagation or composites).
5. **No Generic Fixes**: Strictly warns against generic stop-drilling, grinding, metal patching, welding, or bonding without OEM engineering disposition.
6. **Regulatory Clearance**: Grounding or flight clearance decisions remain strictly with authorized maintenance personnel.

---

### 3.3 Unsupported Defect Classes & Domain Screening Handling

- **Validated Classes**: The trained VGG16 model strictly supports `dent` and `crack`.
- **Unsupported Classes**: Requests for corrosion, delamination, buckling, lightning strikes, or fastener shear return an explicit unsupported class notice rather than hallucinating predictions.
- **Out-of-Domain & Benign Aircraft Screening**: Stage A (CLIP contrastive OOD screening) and Stage B (damage presence detection) safely reject non-aircraft or intact aircraft photos without computing spurious defect activations.

---

## 4. Architecture & API Specifications

### 4.1 Backend Service (`src/aircraft_ai/maintenance.py`)

```python
def get_damage_recommendation(
    predicted_class: Optional[str],
    model_score: Optional[float] = None,
    uncertainty_status: str = "confident",
    context: Optional[Any] = None,
    inspection_id: Optional[str] = None,
    uncertainty_reason: Optional[str] = None,
) -> DamageRecommendation:
    ...
```

### 4.2 Endpoints

- `POST /api/analyze`: Accepts image upload and optional JSON `context`. Executes Stage A (domain screening), Stage B (damage presence), Stage C (VGG16 classification), and maintenance recommendation evaluation.
- `POST /api/recommendation`: Re-evaluates maintenance recommendations on the fly when inspectors enter or modify parameters in the UI without re-running neural network inference.
- `POST /api/inspection/report/html`: Generates printable aerospace NDI report records with context, ordered steps, verified references, and signoff blocks.

---

## 5. Verification & Test Suite Summary

A comprehensive 15-scenario test suite was implemented in [test_maintenance.py](file:///d:/Aircraft%20classification/tests/unit/test_maintenance.py):

| Scenario | Description | Status |
| :--- | :--- | :--- |
| **Scenario 1** | Dent with unknown dimensions -> requires physical measurements | **PASSED** |
| **Scenario 2** | Dent with supplied measurements but no maintenance limits -> doc review required | **PASSED** |
| **Scenario 3** | Dent near structural joint/fastener -> qualified repair engineering required | **PASSED** |
| **Scenario 4** | Suspected crack with uncertain model output -> cautious detailed inspection | **PASSED** |
| **Scenario 5** | Suspected crack with known component info -> procedure-guided NDT selection | **PASSED** |
| **Scenario 6** | Unknown aircraft model -> reported in missing information | **PASSED** |
| **Scenario 7** | Missing maintenance documentation -> standard missing approved data notice | **PASSED** |
| **Scenario 8** | Unsupported defect class -> safe rejection without invented limits | **PASSED** |
| **Scenario 9** | Invalid or missing measurements -> input sanitization & validation | **PASSED** |
| **Scenario 10** | Model inference failure -> safe fallback recommendation | **PASSED** |
| **Scenario 11** | Uncertain or out-of-domain image -> Stage A rejection, limits withheld | **PASSED** |
| **Scenario 12** | Verified maintenance reference retrieval -> FAA AC 43.13-1B verified | **PASSED** |
| **Scenario 13** | Unavailable or inapplicable reference -> marked unverified with warning | **PASSED** |
| **Scenario 14** | Frontend rendering data contract & API recommendation endpoint | **PASSED** |
| **Scenario 15** | Printable inspection report generation & JSON export verification | **PASSED** |

**Full Workspace Test Results**: 62 / 62 tests passing (100% pass rate across unit, integration, and regression suites).
