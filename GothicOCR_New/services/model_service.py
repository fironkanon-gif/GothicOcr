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
)

from services.image_service import ImageService


# ============================================================
# TFLITE RUNTIME
# ============================================================

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        Interpreter = None


class ModelService:
    """
    Handles the GothicOCR TFLite model.

    Model:
        gothic_ocr.tflite

    Input:
        [1, 1024, 1024, 3]
        FLOAT32

    Output:
        [1, 29, 21504]
        FLOAT32

    29 channels:
        4 box values + 25 class scores
    """

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        if Interpreter is None:
            raise ImportError(
                "TFLite Interpreter is not available. "
                "Install tflite-runtime or TensorFlow."
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

        self.interpreter = Interpreter(
            model_path=str(self.model_path)
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self._validate_model()

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    def _validate_model(self):
        """
        Validate the TFLite model against the expected structure.
        """

        if len(self.input_details) != 1:
            raise ValueError(
                f"Expected 1 input tensor, "
                f"found {len(self.input_details)}."
            )

        if len(self.output_details) != 1:
            raise ValueError(
                f"Expected 1 output tensor, "
                f"found {len(self.output_details)}."
            )

        input_shape = tuple(
            self.input_details[0]["shape"]
        )

        expected_input_shape = (
            1,
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
            3,
        )

        if input_shape != expected_input_shape:
            raise ValueError(
                f"Unexpected model input shape: {input_shape}. "
                f"Expected {expected_input_shape}."
            )

        input_dtype = self.input_details[0]["dtype"]

        if input_dtype != np.float32:
            raise ValueError(
                f"Unexpected model input dtype: {input_dtype}. "
                f"Expected float32."
            )

        output_shape = tuple(
            self.output_details[0]["shape"]
        )

        expected_output_shape = (
            1,
            MODEL_OUTPUT_CHANNELS,
            21504,
        )

        if output_shape != expected_output_shape:
            raise ValueError(
                f"Unexpected model output shape: {output_shape}. "
                f"Expected {expected_output_shape}."
            )

        output_dtype = self.output_details[0]["dtype"]

        if output_dtype != np.float32:
            raise ValueError(
                f"Unexpected model output dtype: {output_dtype}. "
                f"Expected float32."
            )

    # ========================================================
    # LETTERBOX
    # ========================================================

    @staticmethod
    def letterbox(
        image: Image.Image,
        new_width: int = MODEL_INPUT_WIDTH,
        new_height: int = MODEL_INPUT_HEIGHT,
        color=(114, 114, 114),
    ):
        """
        Resize an image while preserving its aspect ratio.

        Returns:
            padded_image
            scale
            pad_x
            pad_y
        """

        image = ImageService.to_rgb(image)

        original_width, original_height = image.size

        if original_width <= 0 or original_height <= 0:
            raise ValueError("Invalid image dimensions.")

        scale = min(
            new_width / original_width,
            new_height / original_height,
        )

        resized_width = max(
            1,
            int(round(original_width * scale))
        )

        resized_height = max(
            1,
            int(round(original_height * scale))
        )

        resized = image.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (new_width, new_height),
            color,
        )

        pad_x = (new_width - resized_width) / 2
        pad_y = (new_height - resized_height) / 2

        canvas.paste(
            resized,
            (
                int(round(pad_x)),
                int(round(pad_y)),
            ),
        )

        return canvas, scale, pad_x, pad_y

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    def prepare_input(self, image):
        """
        Prepare image for the model while preserving aspect ratio.
        """

        if isinstance(image, (str, Path)):
            image = ImageService.load_image(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL image or image path."
            )

        image = ImageService.to_rgb(image)

        original_size = image.size

        processed, scale, pad_x, pad_y = self.letterbox(image)

        array = np.asarray(
            processed,
            dtype=np.float32,
        )

        array /= 255.0

        array = np.expand_dims(
            array,
            axis=0,
        )

        expected_shape = (
            1,
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
            3,
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"Prepared input shape {array.shape} "
                f"does not match {expected_shape}."
            )

        return (
            array,
            original_size,
            scale,
            pad_x,
            pad_y,
        )

    # ========================================================
    # INFERENCE
    # ========================================================

    def run_inference(self, input_tensor):
        """
        Run TFLite inference.

        Returns the single model output.
        """

        input_index = self.input_details[0]["index"]

        self.interpreter.set_tensor(
            input_index,
            input_tensor,
        )

        self.interpreter.invoke()

        output_index = self.output_details[0]["index"]

        output = self.interpreter.get_tensor(
            output_index
        )

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        return output

    # ========================================================
    # OUTPUT NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_output(output):
        """
        Normalize model output to:

            [29, 21504]

        regardless of a possible batch dimension.
        """

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        if output.ndim == 3:
            if output.shape[0] != 1:
                raise ValueError(
                    f"Unexpected batch size: {output.shape}"
                )

            output = output[0]

        if output.ndim != 2:
            raise ValueError(
                f"Unexpected output dimensions: {output.shape}"
            )

        if output.shape == (MODEL_OUTPUT_CHANNELS, 21504):
            return output

        if output.shape == (21504, MODEL_OUTPUT_CHANNELS):
            return output.T

        raise ValueError(
            f"Unexpected output shape: {output.shape}. "
            f"Expected "
            f"({MODEL_OUTPUT_CHANNELS}, 21504) "
            f"or "
            f"(21504, {MODEL_OUTPUT_CHANNELS})."
        )

    # ========================================================
    # BOX CONVERSION
    # ========================================================

    @staticmethod
    def xywh_to_xyxy(boxes):
        """
        Convert:
            center_x, center_y, width, height

        to:
            x1, y1, x2, y2
        """

        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        )

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

    # ========================================================
    # IOU
    # ========================================================

    @staticmethod
    def box_iou(box, boxes):
        """
        Calculate IoU between one box and many boxes.
        """

        if len(boxes) == 0:
            return np.array([], dtype=np.float32)

        x1 = np.maximum(
            box[0],
            boxes[:, 0],
        )

        y1 = np.maximum(
            box[1],
            boxes[:, 1],
        )

        x2 = np.minimum(
            box[2],
            boxes[:, 2],
        )

        y2 = np.minimum(
            box[3],
            boxes[:, 3],
        )

        intersection = np.maximum(
            0,
            x2 - x1,
        ) * np.maximum(
            0,
            y2 - y1,
        )

        area_box = max(
            0,
            box[2] - box[0],
        ) * max(
            0,
            box[3] - box[1],
        )

        area_boxes = np.maximum(
            0,
            boxes[:, 2] - boxes[:, 0],
        ) * np.maximum(
            0,
            boxes[:, 3] - boxes[:, 1],
        )

        union = (
            area_box
            + area_boxes
            - intersection
        )

        return intersection / (
            union + 1e-7
        )

    # ========================================================
    # NMS
    # ========================================================

    def nms(
        self,
        boxes,
        scores,
        class_ids,
    ):
        """
        Class-aware Non-Maximum Suppression.
        """

        if len(boxes) == 0:
            return []

        keep = []

        for class_id in np.unique(class_ids):

            indices = np.where(
                class_ids == class_id
            )[0]

            class_scores = scores[indices]

            order = indices[
                np.argsort(
                    class_scores
                )[::-1]
            ]

            while len(order) > 0:

                current = order[0]

                keep.append(int(current))

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

        keep.sort(
            key=lambda i: float(scores[i]),
            reverse=True,
        )

        return keep

    # ========================================================
    # DECODE DETECTIONS
    # ========================================================

    def decode_predictions(
        self,
        output,
        original_size,
        scale,
        pad_x,
        pad_y,
    ):
        """
        Decode YOLO-style output.

        Expected:
            [29, 21504]

        First 4 channels:
            x, y, w, h

        Remaining 25 channels:
            class scores
        """

        output = self.normalize_output(output)

        boxes = output[:4, :].T

        class_scores = output[
            4:4 + NUM_CLASSES,
            :
        ].T

        if class_scores.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} class scores, "
                f"got {class_scores.shape}"
            )

        # Best class for every candidate.
        class_ids = np.argmax(
            class_scores,
            axis=1,
        )

        scores = np.max(
            class_scores,
            axis=1,
        )

        # Remove invalid values.
        valid = np.isfinite(scores)

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        # Confidence filtering.
        keep = scores >= self.confidence_threshold

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if len(boxes) == 0:
            return []

        # Convert xywh -> xyxy.
        boxes = self.xywh_to_xyxy(boxes)

        # ----------------------------------------------------
        # Convert coordinates from letterboxed image
        # back to original image coordinates.
        # ----------------------------------------------------

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y

        boxes /= scale

        original_width, original_height = original_size

        boxes[:, 0] = np.clip(
            boxes[:, 0],
            0,
            original_width,
        )

        boxes[:, 2] = np.clip(
            boxes[:, 2],
            0,
            original_width,
        )

        boxes[:, 1] = np.clip(
            boxes[:, 1],
            0,
            original_height,
        )

        boxes[:, 3] = np.clip(
            boxes[:, 3],
            0,
            original_height,
        )

        # Remove boxes with invalid dimensions.
        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes) == 0:
            return []

        # NMS.
        keep = self.nms(
            boxes,
            scores,
            class_ids,
        )

        detections = []

        for index in keep:

            class_id = int(
                class_ids[index]
            )

            if not (
                0 <= class_id < NUM_CLASSES
            ):
                continue

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(
                    scores[index]
                ),
                "box": [
                    float(boxes[index, 0]),
                    float(boxes[index, 1]),
                    float(boxes[index, 2]),
                    float(boxes[index, 3]),
                ],
            })

        return detections

    # ========================================================
    # SORT DETECTIONS
    # ========================================================

    @staticmethod
    def sort_detections(detections):
        """
        Sort characters from left to right,
        then top to bottom.

        This gives a basic reading order.
        """

        return sorted(
            detections,
            key=lambda d: (
                d["box"][1],
                d["box"][0],
            ),
        )

    # ========================================================
    # GROUP INTO LINES
    # ========================================================

    @staticmethod
    def group_into_lines(
        detections,
        y_tolerance=30,
    ):
        """
        Group detected characters into text lines.
        """

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

            y_center = (
                detection["box"][1]
                + detection["box"][3]
            ) / 2

            placed = False

            for line in lines:

                line_y = np.mean([
                    (
                        item["box"][1]
                        + item["box"][3]
                    ) / 2
                    for item in line
                ])

                if abs(
                    y_center - line_y
                ) <= y_tolerance:

                    line.append(detection)
                    placed = True
                    break

            if not placed:
                lines.append([detection])

        # Sort each line left -> right.
        for line in lines:
            line.sort(
                key=lambda d: d["box"][0]
            )

        # Sort lines top -> bottom.
        lines.sort(
            key=lambda line: min(
                d["box"][1]
                for d in line
            )
        )

        return lines

    # ========================================================
    # TEXT DECODING
    # ========================================================

    @staticmethod
    def detections_to_text(detections):
        """
        Convert detections to a Gothic Unicode string.
        """

        return "".join(
            d["class_name"]
            for d in detections
        )

    @staticmethod
    def lines_to_text(lines):
        """
        Convert grouped lines to multiline text.
        """

        return "\n".join(
            ModelService.detections_to_text(
                line
            )
            for line in lines
        )

    # ========================================================
    # COMPLETE PREDICTION
    # ========================================================

    def predict(
        self,
        image: Union[str, Path, Image.Image],
    ):
        """
        Complete OCR pipeline.

        Returns:
            {
                "text": str,
                "detections": list,
                "lines": list,
            }
        """

        (
            input_tensor,
            original_size,
            scale,
            pad_x,
            pad_y,
        ) = self.prepare_input(image)

        raw_output = self.run_inference(
            input_tensor
        )

        detections = self.decode_predictions(
            raw_output,
            original_size,
            scale,
            pad_x,
            pad_y,
        )

        detections = self.sort_detections(
            detections
        )

        lines = self.group_into_lines(
            detections
        )

        text = self.lines_to_text(
            lines
        )

        return {
            "text": text,
            "detections": detections,
            "lines": lines,
        }

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(self):
        """
        Return model information useful for diagnostics.
        """

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
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "iou_threshold": (
                self.iou_threshold
            ),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GothicOCR ModelService")
    print("=" * 60)

    try:
        service = ModelService()

        info = service.get_model_info()

        print("✅ Model loaded successfully")
        print(
            "Input shape:",
            info["input_shape"]
        )
        print(
            "Input dtype:",
            info["input_dtype"]
        )
        print(
            "Output shape:",
            info["output_shape"]
        )
        print(
            "Output dtype:",
            info["output_dtype"]
        )
        print(
            "Classes:",
            info["num_classes"]
        )

        print("Characters:")
        print(
            "".join(info["class_names"])
        )

    except Exception as error:
        print(
            "❌ ModelService test failed:"
        )
        print(error)
