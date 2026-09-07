
# ============================================================
# GothicOCR - Model Service
# ============================================================

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from utils.constants import (
    MODEL_PATH,
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT,
    NUM_CLASSES,
    CLASS_NAMES,
    MODEL_OUTPUT_CHANNELS,
    MODEL_OUTPUT_DETECTIONS,
)

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        Interpreter = None


class ModelService:

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):

        if Interpreter is None:
            raise ImportError(
                "TFLite Interpreter is not installed."
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.iou_threshold = float(
            iou_threshold
        )

        self.interpreter = Interpreter(
            model_path=str(self.model_path)
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        self._validate_model()

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    def _validate_model(self):

        if len(self.input_details) != 1:
            raise ValueError(
                "Model must have exactly one input."
            )

        if len(self.output_details) != 1:
            raise ValueError(
                "Model must have exactly one output."
            )

        input_shape = tuple(
            self.input_details[0]["shape"]
        )

        expected_input = (
            1,
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
            3,
        )

        if input_shape != expected_input:
            raise ValueError(
                f"Unexpected input shape: {input_shape}"
            )

        if self.input_details[0]["dtype"] != np.float32:
            raise ValueError(
                "Model input must be FLOAT32."
            )

        output_shape = tuple(
            self.output_details[0]["shape"]
        )

        expected_output = (
            1,
            MODEL_OUTPUT_CHANNELS,
            MODEL_OUTPUT_DETECTIONS,
        )

        if output_shape != expected_output:
            raise ValueError(
                f"Unexpected output shape: {output_shape}"
            )

        if self.output_details[0]["dtype"] != np.float32:
            raise ValueError(
                "Model output must be FLOAT32."
            )

    # --------------------------------------------------------
    # LETTERBOX
    # --------------------------------------------------------

    @staticmethod
    def letterbox(image):

        image = image.convert("RGB")

        width, height = image.size

        scale = min(
            MODEL_INPUT_WIDTH / width,
            MODEL_INPUT_HEIGHT / height,
        )

        new_width = max(
            1,
            int(round(width * scale))
        )

        new_height = max(
            1,
            int(round(height * scale))
        )

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (
                MODEL_INPUT_WIDTH,
                MODEL_INPUT_HEIGHT,
            ),
            (114, 114, 114),
        )

        pad_x = (
            MODEL_INPUT_WIDTH - new_width
        ) / 2

        pad_y = (
            MODEL_INPUT_HEIGHT - new_height
        ) / 2

        canvas.paste(
            resized,
            (
                int(round(pad_x)),
                int(round(pad_y)),
            ),
        )

        return canvas, scale, pad_x, pad_y

    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    def prepare_input(self, image):

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be PIL Image or path."
            )

        image = image.convert("RGB")

        original_size = image.size

        image, scale, pad_x, pad_y = (
            self.letterbox(image)
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        array /= 255.0

        array = np.expand_dims(
            array,
            axis=0,
        )

        return (
            array,
            original_size,
            scale,
            pad_x,
            pad_y,
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    def run_inference(self, input_tensor):

        input_index = (
            self.input_details[0]["index"]
        )

        self.interpreter.set_tensor(
            input_index,
            input_tensor,
        )

        self.interpreter.invoke()

        output_index = (
            self.output_details[0]["index"]
        )

        return self.interpreter.get_tensor(
            output_index
        )

    # --------------------------------------------------------
    # NORMALIZE OUTPUT
    # --------------------------------------------------------

    @staticmethod
    def normalize_output(output):

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        if output.ndim == 3:
            output = output[0]

        if output.shape == (
            MODEL_OUTPUT_CHANNELS,
            MODEL_OUTPUT_DETECTIONS,
        ):
            return output

        if output.shape == (
            MODEL_OUTPUT_DETECTIONS,
            MODEL_OUTPUT_CHANNELS,
        ):
            return output.T

        raise ValueError(
            f"Unexpected output shape: {output.shape}"
        )

    # --------------------------------------------------------
    # XYWH -> XYXY
    # --------------------------------------------------------

    @staticmethod
    def xywh_to_xyxy(boxes):

        result = np.empty_like(boxes)

        result[:, 0] = (
            boxes[:, 0] - boxes[:, 2] / 2
        )

        result[:, 1] = (
            boxes[:, 1] - boxes[:, 3] / 2
        )

        result[:, 2] = (
            boxes[:, 0] + boxes[:, 2] / 2
        )

        result[:, 3] = (
            boxes[:, 1] + boxes[:, 3] / 2
        )

        return result

    # --------------------------------------------------------
    # IOU
    # --------------------------------------------------------

    @staticmethod
    def box_iou(box, boxes):

        if len(boxes) == 0:
            return np.array(
                [],
                dtype=np.float32
            )

        x1 = np.maximum(
            box[0],
            boxes[:, 0]
        )

        y1 = np.maximum(
            box[1],
            boxes[:, 1]
        )

        x2 = np.minimum(
            box[2],
            boxes[:, 2]
        )

        y2 = np.minimum(
            box[3],
            boxes[:, 3]
        )

        inter = (
            np.maximum(0, x2 - x1)
            *
            np.maximum(0, y2 - y1)
        )

        area1 = (
            max(0, box[2] - box[0])
            *
            max(0, box[3] - box[1])
        )

        area2 = (
            np.maximum(
                0,
                boxes[:, 2] - boxes[:, 0]
            )
            *
            np.maximum(
                0,
                boxes[:, 3] - boxes[:, 1]
            )
        )

        union = area1 + area2 - inter

        return inter / (union + 1e-7)

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    def nms(
        self,
        boxes,
        scores,
        class_ids,
    ):

        keep = []

        for class_id in np.unique(class_ids):

            indices = np.where(
                class_ids == class_id
            )[0]

            order = indices[
                np.argsort(
                    scores[indices]
                )[::-1]
            ]

            while len(order):

                current = order[0]

                keep.append(
                    int(current)
                )

                if len(order) == 1:
                    break

                remaining = order[1:]

                ious = self.box_iou(
                    boxes[current],
                    boxes[remaining],
                )

                order = remaining[
                    ious < self.iou_threshold
                ]

        return sorted(
            keep,
            key=lambda i: scores[i],
            reverse=True,
        )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    def decode_predictions(
        self,
        output,
        original_size,
        scale,
        pad_x,
        pad_y,
    ):

        output = self.normalize_output(
            output
        )

        boxes = output[:4].T

        class_scores = output[
            4:4 + NUM_CLASSES
        ].T

        class_ids = np.argmax(
            class_scores,
            axis=1,
        )

        scores = np.max(
            class_scores,
            axis=1,
        )

        valid = (
            np.isfinite(scores)
            &
            (scores >= self.confidence_threshold)
        )

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes) == 0:
            return []

        boxes = self.xywh_to_xyxy(
            boxes
        )

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y

        boxes /= scale

        width, height = original_size

        boxes[:, 0] = np.clip(
            boxes[:, 0],
            0,
            width,
        )

        boxes[:, 2] = np.clip(
            boxes[:, 2],
            0,
            width,
        )

        boxes[:, 1] = np.clip(
            boxes[:, 1],
            0,
            height,
        )

        boxes[:, 3] = np.clip(
            boxes[:, 3],
            0,
            height,
        )

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            &
            (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes) == 0:
            return []

        keep = self.nms(
            boxes,
            scores,
            class_ids,
        )

        detections = []

        for i in keep:

            class_id = int(
                class_ids[i]
            )

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(scores[i]),
                "box": [
                    float(boxes[i, 0]),
                    float(boxes[i, 1]),
                    float(boxes[i, 2]),
                    float(boxes[i, 3]),
                ],
            })

        return detections

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    @staticmethod
    def sort_detections(
        detections
    ):

        return sorted(
            detections,
            key=lambda d: (
                d["box"][1],
                d["box"][0],
            ),
        )

    # --------------------------------------------------------
    # GROUP LINES
    # --------------------------------------------------------

    @staticmethod
    def group_into_lines(
        detections,
        y_tolerance=30,
    ):

        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda d: (
                d["box"][1],
                d["box"][0],
            ),
        )

        lines = []

        for detection in detections:

            y = (
                detection["box"][1]
                +
                detection["box"][3]
            ) / 2

            found = False

            for line in lines:

                line_y = np.mean([
                    (
                        d["box"][1]
                        +
                        d["box"][3]
                    ) / 2
                    for d in line
                ])

                if abs(y - line_y) <= y_tolerance:
                    line.append(detection)
                    found = True
                    break

            if not found:
                lines.append(
                    [detection]
                )

        for line in lines:
            line.sort(
                key=lambda d: d["box"][0]
            )

        lines.sort(
            key=lambda line: min(
                d["box"][1]
                for d in line
            )
        )

        return lines

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    @staticmethod
    def detections_to_text(
        detections
    ):

        return "".join(
            d["class_name"]
            for d in detections
        )

    @staticmethod
    def lines_to_text(lines):

        return "\n".join(
            ModelService.detections_to_text(
                line
            )
            for line in lines
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(self, image):

        (
            input_tensor,
            original_size,
            scale,
            pad_x,
            pad_y,
        ) = self.prepare_input(image)

        output = self.run_inference(
            input_tensor
        )

        detections = (
            self.decode_predictions(
                output,
                original_size,
                scale,
                pad_x,
                pad_y,
            )
        )

        detections = (
            self.sort_detections(
                detections
            )
        )

        lines = (
            self.group_into_lines(
                detections
            )
        )

        text = self.lines_to_text(
            lines
        )

        return {
            "text": text,
            "detections": detections,
            "lines": lines,
        }

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    def get_model_info(self):

        return {
            "model_path": str(
                self.model_path
            ),
            "input_shape": list(
                self.input_details[0]["shape"]
            ),
            "input_dtype": str(
                self.input_details[0]["dtype"]
            ),
            "output_shape": list(
                self.output_details[0]["shape"]
            ),
            "output_dtype": str(
                self.output_details[0]["dtype"]
            ),
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
        }


print("✅ GothicOCR ModelService loaded")
