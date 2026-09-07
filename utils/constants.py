
# ============================================================
# GothicOCR - Constants
# ============================================================

from pathlib import Path


# المشروع الحقيقي هو المجلد الأب لمجلد utils
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "gothic_ocr.tflite"


MODEL_INPUT_WIDTH = 1024
MODEL_INPUT_HEIGHT = 1024

NUM_CLASSES = 25

MODEL_OUTPUT_CHANNELS = 29
MODEL_OUTPUT_DETECTIONS = 21504


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


DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        "CLASS_NAMES and NUM_CLASSES do not match."
    )


if MODEL_OUTPUT_CHANNELS != 4 + NUM_CLASSES:
    raise ValueError(
        "MODEL_OUTPUT_CHANNELS must equal "
        "4 + NUM_CLASSES."
    )
