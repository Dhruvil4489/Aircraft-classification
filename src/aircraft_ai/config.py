"""Configuration module for aircraft damage classification and captioning."""

import os
from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_ROOT / "models"
RESULTS_DIR = ARTIFACTS_ROOT / "results"
LOGS_DIR = ARTIFACTS_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_ROOT, ARTIFACTS_ROOT, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URL = os.getenv(
    "DATASET_URL",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar",
)
TAR_FILENAME = "aircraft_damage_dataset_v1.tar"
DATASET_DIR_NAME = "aircraft_damage_dataset_v1"
DATASET_DIR = DATA_ROOT / DATASET_DIR_NAME

# Expected split paths
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"
SPLITS = ["train", "valid", "test"]
CLASSES = ["crack", "dent"]
CLASS_INDICES = {"crack": 0, "dent": 1}
INVERTED_CLASS_INDICES = {0: "crack", 1: "dent"}

# Academic constants strictly following notebook specifications
SEED_VALUE = int(os.getenv("SEED", "42"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
N_EPOCHS = int(os.getenv("EPOCHS", "5"))
IMG_ROWS = 224
IMG_COLS = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
CLASSIFICATION_THRESHOLD = 0.5

# Model persistence paths
MODEL_SAVE_PATH = MODELS_DIR / "vgg16_aircraft_damage.keras"
CLASS_INDICES_PATH = RESULTS_DIR / "class_indices.json"
TRAINING_CONFIG_PATH = RESULTS_DIR / "training_config.json"
HISTORY_PATH = RESULTS_DIR / "history.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
DATASET_REPORT_PATH = RESULTS_DIR / "dataset_report.json"
BLIP_EXAMPLES_PATH = RESULTS_DIR / "blip_examples.json"

# Plot image paths
TRAINING_LOSS_PLOT_PATH = RESULTS_DIR / "training_loss.png"
VALIDATION_LOSS_PLOT_PATH = RESULTS_DIR / "validation_loss.png"
ACCURACY_CURVE_PLOT_PATH = RESULTS_DIR / "accuracy_curve.png"
CONFUSION_MATRIX_PLOT_PATH = RESULTS_DIR / "confusion_matrix.png"
TASK7_PREDICTION_PLOT_PATH = RESULTS_DIR / "task7_prediction.png"
TEST_PREDICTIONS_GRID_PATH = RESULTS_DIR / "test_predictions_grid.png"

# BLIP Configuration
BLIP_MODEL_ID = os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
BLIP_DEVICE = os.getenv("BLIP_DEVICE", "cpu")
CAPTION_PROMPT = "This is a picture of"
SUMMARY_PROMPT = "This is a detailed photo showing"

# API & Security
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FIGURE_NAMES = {
    "training_loss.png",
    "validation_loss.png",
    "accuracy_curve.png",
    "confusion_matrix.png",
    "task7_prediction.png",
    "test_predictions_grid.png",
}

# Safety disclaimer
SAFETY_DISCLAIMER = "Model output for educational purposes only; not a certified aviation safety assessment."

# Domain Screening & Out-of-Distribution Rejection (CLIP Zero-Shot)
DOMAIN_SCREENING_MODEL_ID = os.getenv("DOMAIN_SCREENING_MODEL_ID", "openai/clip-vit-base-patch32")
DOMAIN_ACCEPTANCE_THRESHOLD = float(os.getenv("DOMAIN_ACCEPTANCE_THRESHOLD", "0.50"))
UNCERTAINTY_MARGIN = float(os.getenv("UNCERTAINTY_MARGIN", "0.12"))

IN_DOMAIN_PROMPTS = [
    "a photo of an aircraft exterior fuselage surface showing damage or defect",
    "a close-up technical inspection photo of an aircraft metal panel with crack or dent",
    "an airplane wing, tail, or metallic skin panel during structural inspection",
]

OUT_OF_DOMAIN_PROMPTS = [
    "a handwritten signature or cursive ink writing on plain paper",
    "a printed text document, letter, invoice, contract, or receipt",
    "a photograph of a person, human face, or portrait",
    "a photograph of an automobile car, truck, or bicycle",
    "an animal, dog, cat, or wildlife",
    "a natural landscape, mountains, forest, ocean, or scenic sky",
    "a digital diagram, screenshot, user interface, or cartoon drawing",
]

# Supported Analysis Outcomes
STATUS_COMPLETED = "COMPLETED"
STATUS_UNSUPPORTED = "UNSUPPORTED_IMAGE"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_FAILED = "ANALYSIS_FAILED"

# P2 Flags (Optional enhancements, default OFF)
ENABLE_DATA_AUGMENTATION = os.getenv("ENABLE_DATA_AUGMENTATION", "false").lower() == "true"
ENABLE_FINE_TUNING = os.getenv("ENABLE_FINE_TUNING", "false").lower() == "true"
ENABLE_CLASS_WEIGHTS = os.getenv("ENABLE_CLASS_WEIGHTS", "false").lower() == "true"

