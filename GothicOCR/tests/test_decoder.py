# ============================================================
# GOTHIC OCR — TEXT DECODER TESTS
# ============================================================

from pathlib import Path
import sys

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


# ============================================================
# IMPORT
# ============================================================

from services.text_decoder import TextDecoder


# ============================================================
# PATHS
# ============================================================

LABELS = (
    ROOT
    / "data"
    / "labels.json"
)


# ============================================================
# TEST LABELS
# ============================================================

def test_labels():

    decoder = TextDecoder(
        LABELS
    )

    assert len(
        decoder.labels
    ) == 25

    assert all(
        isinstance(label, str)
        for label in decoder.labels
    )

    print(
        "PASS: labels loaded"
    )


# ============================================================
# TEST IOU
# ============================================================

def test_iou():

    decoder = TextDecoder(
        LABELS
    )

    # Identical boxes → IoU = 1
    value = decoder.iou(
        (0, 0, 100, 100),
        (0, 0, 100, 100),
    )

    assert np.isclose(
        value,
        1.0,
    )

    # No overlap → IoU = 0
    value = decoder.iou(
        (0, 0, 10, 10),
        (20, 20, 30, 30),
    )

    assert np.isclose(
        value,
        0.0,
    )

    print(
        "PASS: IoU"
    )


# ============================================================
# TEST NMS
# ============================================================

def test_nms():

    decoder = TextDecoder(
        LABELS
    )

    detections = [
        {
            "class_id": 0,
            "letter": decoder.labels[0],
            "score": 0.90,
            "box": (
                10.0,
                10.0,
                100.0,
                100.0,
            ),
        },
        {
            "class_id": 0,
            "letter": decoder.labels[0],
            "score": 0.80,
            "box": (
                12.0,
                12.0,
                98.0,
                98.0,
            ),
        },
    ]

    result = decoder.nms(
        detections
    )

    # Same class + heavy overlap
    # → keep only highest confidence
    assert len(result) == 1

    assert np.isclose(
        result[0]["score"],
        0.90,
    )

    print(
        "PASS: class-aware NMS"
    )


# ============================================================
# TEST DIFFERENT CLASSES
# ============================================================

def test_nms_different_classes():

    decoder = TextDecoder(
        LABELS
    )

    detections = [
        {
            "class_id": 0,
            "letter": decoder.labels[0],
            "score": 0.90,
            "box": (
                10.0,
                10.0,
                100.0,
                100.0,
            ),
        },
        {
            "class_id": 1,
            "letter": decoder.labels[1],
            "score": 0.80,
            "box": (
                12.0,
                12.0,
                98.0,
                98.0,
            ),
        },
    ]

    result = decoder.nms(
        detections
    )

    # Different classes must survive NMS
    assert len(result) == 2

    print(
        "PASS: different classes preserved"
    )


# ============================================================
# CREATE EMPTY YOLO OUTPUT
# ============================================================

def empty_output():

    return np.zeros(
        (
            1,
            29,
            21504,
        ),
        dtype=np.float32,
    )


# ============================================================
# TEST YOLO OUTPUT SHAPE
# ============================================================

def test_decode_empty_output():

    decoder = TextDecoder(
        LABELS
    )

    output = empty_output()

    result = decoder.decode(
        output=output,
        original_width=1024,
        original_height=1024,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )

    assert result["text"] == ""

    assert result["detections"] == []

    assert result["lines"] == []

    print(
        "PASS: empty YOLO output"
    )


# ============================================================
# TEST TRANSPOSED OUTPUT
# ============================================================

def test_decode_transposed_output():

    decoder = TextDecoder(
        LABELS
    )

    output = empty_output()

    transposed = np.transpose(
        output,
        (
            0,
            2,
            1,
        ),
    )

    result = decoder.decode(
        output=transposed,
        original_width=1024,
        original_height=1024,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )

    assert result["text"] == ""

    assert result["detections"] == []

    print(
        "PASS: transposed YOLO output"
    )


# ============================================================
# TEST BOX CONVERSION
# ============================================================

def test_box_conversion():

    decoder = TextDecoder(
        LABELS
    )

    # Normalized coordinates
    x, y, w, h = decoder._box_to_canvas(
        0.5,
        0.5,
        0.25,
        0.25,
    )

    assert np.isclose(
        x,
        512.0,
    )

    assert np.isclose(
        y,
        512.0,
    )

    assert np.isclose(
        w,
        256.0,
    )

    assert np.isclose(
        h,
        256.0,
    )

    # Pixel coordinates
    x, y, w, h = decoder._box_to_canvas(
        512.0,
        512.0,
        256.0,
        256.0,
    )

    assert np.isclose(
        x,
        512.0,
    )

    assert np.isclose(
        y,
        512.0,
    )

    assert np.isclose(
        w,
        256.0,
    )

    assert np.isclose(
        h,
        256.0,
    )

    print(
        "PASS: box coordinate conversion"
    )


# ============================================================
# TEST LINE GROUPING
# ============================================================

def test_line_grouping():

    decoder = TextDecoder(
        LABELS
    )

    detections = [
        {
            "class_id": 0,
            "letter": decoder.labels[0],
            "score": 0.90,
            "box": (
                10.0,
                10.0,
                30.0,
                40.0,
            ),
        },
        {
            "class_id": 1,
            "letter": decoder.labels[1],
            "score": 0.90,
            "box": (
                35.0,
                11.0,
                55.0,
                41.0,
            ),
        },
        {
            "class_id": 2,
            "letter": decoder.labels[2],
            "score": 0.90,
            "box": (
                10.0,
                100.0,
                30.0,
                130.0,
            ),
        },
    ]

    lines = decoder._group_lines(
        detections
    )

    assert len(lines) == 2

    assert len(
        lines[0]["items"]
    ) == 2

    assert len(
        lines[1]["items"]
    ) == 1

    print(
        "PASS: line grouping"
    )


# ============================================================
# RUN ALL TESTS
# ============================================================

def main():

    print("=" * 60)
    print("GOTHIC OCR — TEXT DECODER TESTS")
    print("=" * 60)

    test_labels()
    test_iou()
    test_nms()
    test_nms_different_classes()
    test_decode_empty_output()
    test_decode_transposed_output()
    test_box_conversion()
    test_line_grouping()

    print()
    print("=" * 60)
    print("ALL DECODER TESTS PASSED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
