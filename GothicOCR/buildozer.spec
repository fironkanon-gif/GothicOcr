# ============================================================
# GOTHIC OCR — TEXT DECODER
# ============================================================

from pathlib import Path
import json
import numpy as np


class TextDecoder:

    INPUT_SIZE = 1024
    NUM_CLASSES = 25

    CONFIDENCE_THRESHOLD = 0.05
    NMS_IOU_THRESHOLD = 0.45

    def __init__(
        self,
        labels_path=None,
        confidence_threshold=None,
        nms_iou_threshold=None,
    ):

        if labels_path is None:
            labels_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "labels.json"
            )

        self.labels_path = Path(labels_path)

        self.labels = self._load_labels()

        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.CONFIDENCE_THRESHOLD
        )

        self.nms_iou_threshold = (
            nms_iou_threshold
            if nms_iou_threshold is not None
            else self.NMS_IOU_THRESHOLD
        )

    # ========================================================
    # LOAD LABELS
    # ========================================================

    def _load_labels(self):

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}"
            )

        with open(
            self.labels_path,
            "r",
            encoding="utf-8"
        ) as file:

            labels = json.load(file)

        if not isinstance(labels, list):
            raise ValueError(
                "labels.json must contain a list."
            )

        if len(labels) != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} labels, "
                f"found {len(labels)}."
            )

        return labels

    # ========================================================
    # IOU
    # ========================================================

    @staticmethod
    def iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * \
                 max(0.0, ay2 - ay1)

        area_b = max(0.0, bx2 - bx1) * \
                 max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    # ========================================================
    # CLASS-AWARE NMS
    # ========================================================

    def nms(self, detections):

        if not detections:
            return []

        class_groups = {}

        for detection in detections:

            class_id = detection["class_id"]

            class_groups.setdefault(
                class_id,
                []
            ).append(detection)

        kept = []

        for class_id, group in class_groups.items():

            group = sorted(
                group,
                key=lambda d: d["score"],
                reverse=True
            )

            while group:

                best = group.pop(0)

                kept.append(best)

                remaining = []

                for other in group:

                    overlap = self.iou(
                        best["box"],
                        other["box"]
                    )

                    if overlap < self.nms_iou_threshold:
                        remaining.append(other)

                group = remaining

        return kept

    # ========================================================
    # DECODE RAW OUTPUT
    # ========================================================

    def decode(
        self,
        output,
        original_width,
        original_height,
        scale,
        pad_x,
        pad_y,
    ):
        """
        Decode YOLO-style TFLite output.

        Expected:
            output shape = [1, 29, 21504]
            or
            output shape = [29, 21504]

        Box format:
            normalized top-left XYWH

        Returns:
            {
                "text": str,
                "detections": list,
                "lines": list
            }
        """

        output = np.asarray(output)

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        if output.ndim == 3:

            if output.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size 1, got {output.shape}"
                )

            output = output[0]

        if output.ndim != 2:
            raise ValueError(
                f"Unexpected output shape: {output.shape}"
            )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if output.shape[0] == 29:

            predictions = output

        elif output.shape[1] == 29:

            predictions = output.T

        else:

            raise ValueError(
                f"Expected 29 channels, got {output.shape}"
            )

        boxes = predictions[:4]
        class_scores = predictions[4:29]

        # ----------------------------------------------------
        # Best class
        # ----------------------------------------------------

        scores = np.max(
            class_scores,
            axis=0
        )

        class_ids = np.argmax(
            class_scores,
            axis=0
        )

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        positions = np.where(
            scores >= self.confidence_threshold
        )[0]

        detections = []

        # ----------------------------------------------------
        # Decode boxes
        # ----------------------------------------------------

        for position in positions:

            x = float(
                boxes[0, position]
            )

            y = float(
                boxes[1, position]
            )

            w = float(
                boxes[2, position]
            )

            h = float(
                boxes[3, position]
            )

            score = float(
                scores[position]
            )

            class_id = int(
                class_ids[position]
            )

            if not (
                0 <= class_id < len(self.labels)
            ):
                continue

            # ------------------------------------------------
            # Normalized → 1024 canvas
            # ------------------------------------------------

            canvas_x1 = x * self.INPUT_SIZE
            canvas_y1 = y * self.INPUT_SIZE

            canvas_x2 = (
                canvas_x1 +
                w * self.INPUT_SIZE
            )

            canvas_y2 = (
                canvas_y1 +
                h * self.INPUT_SIZE
            )

            # ------------------------------------------------
            # Remove letterbox
            # ------------------------------------------------

            x1 = (
                canvas_x1 - pad_x
            ) / scale

            y1 = (
                canvas_y1 - pad_y
            ) / scale

            x2 = (
                canvas_x2 - pad_x
            ) / scale

            y2 = (
                canvas_y2 - pad_y
            ) / scale

            # ------------------------------------------------
            # Clip to original image
            # ------------------------------------------------

            x1 = float(
                np.clip(
                    x1,
                    0,
                    original_width
                )
            )

            y1 = float(
                np.clip(
                    y1,
                    0,
                    original_height
                )
            )

            x2 = float(
                np.clip(
                    x2,
                    0,
                    original_width
                )
            )

            y2 = float(
                np.clip(
                    y2,
                    0,
                    original_height
                )
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # Detection
            # ------------------------------------------------

            detections.append({
                "class_id": class_id,

                "letter": self.labels[class_id],

                "score": score,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            })

        # ====================================================
        # NMS
        # ====================================================

        detections = self.nms(
            detections
        )

        # ====================================================
        # LINE GROUPING
        # ====================================================

        lines = self._group_lines(
            detections
        )

        # ====================================================
        # TEXT
        # ====================================================

        text_lines = []

        for line in lines:

            items = line["items"]

            items.sort(
                key=lambda d: d["box"][0]
            )

            line_text = "".join(
                d["letter"]
                for d in items
            )

            if line_text:
                text_lines.append(
                    line_text
                )

        text = "\n".join(
            text_lines
        )

        return {
            "text": text,
            "detections": detections,
            "lines": lines,
        }

    # ========================================================
    # GROUP DETECTIONS INTO LINES
    # ========================================================

    def _group_lines(self, detections):

        if not detections:
            return []

        vertical = sorted(
            detections,
            key=lambda d: (
                d["box"][1] +
                d["box"][3]
            ) / 2
        )

        lines = []

        for detection in vertical:

            x1, y1, x2, y2 = \
                detection["box"]

            center_y = (
                y1 + y2
            ) / 2

            height = max(
                1.0,
                y2 - y1
            )

            added = False

            for line in lines:

                line_items = line["items"]

                line_center_y = \
                    line["center_y"]

                average_height = np.mean([
                    max(
                        1.0,
                        d["box"][3] -
                        d["box"][1]
                    )
                    for d in line_items
                ])

                tolerance = max(
                    height * 0.50,
                    average_height * 0.60,
                    8.0
                )

                if abs(
                    center_y -
                    line_center_y
                ) <= tolerance:

                    line_items.append(
                        detection
                    )

                    line["center_y"] = float(
                        np.mean([
                            (
                                d["box"][1] +
                                d["box"][3]
                            ) / 2
                            for d in line_items
                        ])
                    )

                    added = True
                    break

            if not added:

                lines.append({
                    "center_y": center_y,
                    "items": [
                        detection
                    ]
                })

        # ----------------------------------------------------
        # Sort lines top → bottom
        # ----------------------------------------------------

        lines.sort(
            key=lambda line:
                line["center_y"]
        )

        # ----------------------------------------------------
        # Sort characters left → right
        # ----------------------------------------------------

        for line in lines:

            line["items"].sort(
                key=lambda d:
                    d["box"][0]
            )

        return lines

    # ========================================================
    # SIMPLE TEXT ONLY
    # ========================================================

    def decode_text(
        self,
        output,
        original_width,
        original_height,
        scale,
        pad_x,
        pad_y,
    ):

        result = self.decode(
            output,
            original_width,
            original_height,
            scale,
            pad_x,
            pad_y,
        )

        return result["text"]