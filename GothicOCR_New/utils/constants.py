# ============================================================
# GothicOCR - Constants
# ============================================================
# Central configuration shared by the application.
# ============================================================

from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "gothic_ocr.tflite"


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

MODEL_INPUT_WIDTH = 1024
MODEL_INPUT_HEIGHT = 1024

NUM_CLASSES = 25

# YOLO-style output:
# [batch, 4 box values + 25 class scores, detections]
MODEL_OUTPUT_CHANNELS = 29
MODEL_OUTPUT_DETECTIONS = 21504


# ------------------------------------------------------------
# GOTHIC CHARACTER CLASSES
# ------------------------------------------------------------

CLASS_NAMES = [
    "𐌰",
    "𐌱",
    "𐌲",
    "𐌳",
    "𐌴",
    "𐌵",
    "𐌶",
    "𐌷",
    "𐌸",
    "𐌹",
    "𐌺",
    "𐌻",
    "𐌼",
    "𐌽",
    "𐌾",
    "𐌿",
    "𐍀",
    "𐍂",
    "𐍃",
    "𐍄",
    "𐍅",
    "𐍆",
    "𐍇",
    "𐍈",
    "𐍉",
]


# ------------------------------------------------------------
# IMAGE CONFIGURATION
# ------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MODEL_INPUT_SIZE = (
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT,
)


# ------------------------------------------------------------
# DETECTION CONFIGURATION
# ------------------------------------------------------------

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        f"CLASS_NAMES contains {len(CLASS_NAMES)} classes, "
        f"but NUM_CLASSES is {NUM_CLASSES}."
    )

if MODEL_OUTPUT_CHANNELS != 4 + NUM_CLASSES:
    raise ValueError(
        "MODEL_OUTPUT_CHANNELS does not match "
        "4 box values + NUM_CLASSES."
    )
