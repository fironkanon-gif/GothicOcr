# ============================================================
# GOTHIC OCR — MODEL SERVICE
# ============================================================

from pathlib import Path

import numpy as np

from services.image_service import ImageService
from services.text_decoder import TextDecoder


class GothicOCR:
    """
    Android-compatible Gothic OCR inference service.

    Pipeline:

        image
          ↓
        ImageService
          ↓
        NCHW float32
          ↓
        tflite_runtime.Interpreter
          ↓
        NumPy output
          ↓
        TextDecoder
    """

    INPUT_SHAPE = (1, 3, 1024, 1024)
    OUTPUT_SHAPE = (1, 29, 21504)

    INPUT_SIZE = 1024

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.interpreter = None

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # ----------------------------------------------------
        # TensorFlow Lite Runtime
        # ----------------------------------------------------
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception as error:
            raise RuntimeError(
                "tflite-runtime is required for GothicOCR Android inference."
            ) from error

        # ----------------------------------------------------
        # Create interpreter
        # ----------------------------------------------------
        try:
            self.interpreter = Interpreter(
                model_path=str(self.model_path)
            )

            self.interpreter.allocate_tensors()

        except Exception as error:
            self.interpreter = None

            raise RuntimeError(
                "Failed to create TensorFlow Lite interpreter: "
                f"{error}"
            ) from error

        # ----------------------------------------------------
        # Tensor metadata
        # ----------------------------------------------------
        try:
            self.input_details = (
                self.interpreter.get_input_details()
            )

            self.output_details = (
                self.interpreter.get_output_details()
            )

        except Exception as error:
            self.close()

            raise RuntimeError(
                "Failed to inspect TensorFlow Lite tensors: "
                f"{error}"
            ) from error

        if len(self.input_details) != 1:
            self.close()
            raise RuntimeError(
                "Expected exactly one input tensor, "
                f"found {len(self.input_details)}."
            )

        if len(self.output_details) != 1:
            self.close()
            raise RuntimeError(
                "Expected exactly one output tensor, "
                f"found {len(self.output_details)}."
            )

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------
        input_detail = self.input_details[0]
        output_detail = self.output_details[0]

        input_shape = tuple(
            int(value)
            for value in input_detail["shape"]
        )

        output_shape = tuple(
            int(value)
            for value in output_detail["shape"]
        )

        input_dtype = np.dtype(
            input_detail["dtype"]
        )

        output_dtype = np.dtype(
            output_detail["dtype"]
        )

        if input_shape != self.INPUT_SHAPE:
            self.close()

            raise ValueError(
                "Unexpected model input shape: "
                f"{input_shape}. "
                f"Expected {self.INPUT_SHAPE}."
            )

        if output_shape != self.OUTPUT_SHAPE:
            self.close()

            raise ValueError(
                "Unexpected model output shape: "
                f"{output_shape}. "
                f"Expected {self.OUTPUT_SHAPE}."
            )

        if input_dtype != np.dtype(np.float32):
            self.close()

            raise ValueError(
                "Unsupported input tensor dtype: "
                f"{input_dtype}. Expected float32."
            )

        if output_dtype != np.dtype(np.float32):
            self.close()

            raise ValueError(
                "Unsupported output tensor dtype: "
                f"{output_dtype}. Expected float32."
            )

        # ----------------------------------------------------
        # Services
        # ----------------------------------------------------
        self.image_service = ImageService(
            target_size=self.INPUT_SIZE
        )

        self.decoder = TextDecoder()

        self.input_shape = input_shape
        self.output_shape = output_shape

        # ----------------------------------------------------
        # Sizes
        # ----------------------------------------------------
        self.input_elements = int(
            np.prod(self.INPUT_SHAPE)
        )

        self.output_elements = int(
            np.prod(self.OUTPUT_SHAPE)
        )

        self.input_bytes = (
            self.input_elements
            * np.dtype(np.float32).itemsize
        )

        self.output_bytes = (
            self.output_elements
            * np.dtype(np.float32).itemsize
        )

        print("========================================")
        print("GOTHIC OCR MODEL SERVICE")
        print("========================================")
        print("Runtime: tflite_runtime")
        print("Model:", self.model_path)
        print("Input:", self.input_shape)
        print("Output:", self.output_shape)
        print("Input dtype:", input_dtype)
        print("Output dtype:", output_dtype)
        print("Input bytes:", self.input_bytes)
        print("Output bytes:", self.output_bytes)
        print("========================================")

    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    def _prepare_image(self, image):

        if isinstance(image, (str, Path)):

            prepared_input, metadata = (
                self.image_service.load_and_prepare(
                    image
                )
            )

        else:

            array = np.asarray(image)

            if array.ndim != 3:
                raise ValueError(
                    "Image must have shape [H, W, C]. "
                    f"Got {array.shape}."
                )

            if array.shape[2] != 3:
                raise ValueError(
                    "Image must have 3 RGB channels. "
                    f"Got {array.shape}."
                )

            if array.size == 0:
                raise ValueError(
                    "Image is empty."
                )

            if array.dtype != np.uint8:
                array = np.clip(
                    array,
                    0,
                    255
                ).astype(np.uint8)

            prepared_input, metadata = (
                self.image_service.prepare(
                    array
                )
            )

        prepared_input = np.asarray(
            prepared_input,
            dtype=np.float32
        )

        if prepared_input.shape != self.INPUT_SHAPE:
            raise ValueError(
                "Prepared input has wrong shape: "
                f"{prepared_input.shape}. "
                f"Expected {self.INPUT_SHAPE}."
            )

        if not prepared_input.flags["C_CONTIGUOUS"]:
            prepared_input = np.ascontiguousarray(
                prepared_input,
                dtype=np.float32
            )

        if not np.all(
            np.isfinite(prepared_input)
        ):
            raise ValueError(
                "Prepared input contains NaN or Inf."
            )

        required_metadata = (
            "original_width",
            "original_height",
            "scale",
            "pad_x",
            "pad_y",
        )

        for key in required_metadata:
            if key not in metadata:
                raise ValueError(
                    "ImageService metadata is missing "
                    f"'{key}'."
                )

        if float(metadata["scale"]) <= 0:
            raise ValueError(
                "Invalid image scale: "
                f"{metadata['scale']}"
            )

        return prepared_input, metadata

    # ========================================================
    # RUN TFLITE
    # ========================================================

    def _run_inference(self, prepared_input):

        tensor = np.asarray(
            prepared_input,
            dtype=np.float32
        )

        tensor = np.ascontiguousarray(
            tensor,
            dtype=np.float32
        )

        if tensor.shape != self.INPUT_SHAPE:
            raise ValueError(
                "Input tensor shape mismatch: "
                f"{tensor.shape}. "
                f"Expected {self.INPUT_SHAPE}."
            )

        if tensor.nbytes != self.input_bytes:
            raise ValueError(
                "Input byte size mismatch: "
                f"{tensor.nbytes} != {self.input_bytes}"
            )

        if not np.all(
            np.isfinite(tensor)
        ):
            raise ValueError(
                "Input tensor contains NaN or Inf."
            )

        input_index = self.input_details[0]["index"]
        output_index = self.output_details[0]["index"]

        # ----------------------------------------------------
        # Set input
        # ----------------------------------------------------
        try:
            self.interpreter.set_tensor(
                input_index,
                tensor
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to set TensorFlow Lite input tensor: "
                f"{error}"
            ) from error

        # ----------------------------------------------------
        # Invoke
        # ----------------------------------------------------
        try:
            self.interpreter.invoke()

        except Exception as error:
            raise RuntimeError(
                "TensorFlow Lite inference failed: "
                f"{error}"
            ) from error

        # ----------------------------------------------------
        # Get output
        # ----------------------------------------------------
        try:
            output = self.interpreter.get_tensor(
                output_index
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to read TensorFlow Lite output: "
                f"{error}"
            ) from error

        output = np.asarray(
            output,
            dtype=np.float32
        )

        if output.shape != self.OUTPUT_SHAPE:
            raise ValueError(
                "Unexpected output shape: "
                f"{output.shape}. "
                f"Expected {self.OUTPUT_SHAPE}."
            )

        if output.size != self.output_elements:
            raise ValueError(
                "Unexpected output element count: "
                f"{output.size}. "
                f"Expected {self.output_elements}."
            )

        if not np.all(
            np.isfinite(output)
        ):
            raise ValueError(
                "TensorFlow Lite output contains "
                "NaN or Inf."
            )

        return output

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, image):

        prepared_input, metadata = (
            self._prepare_image(image)
        )

        output = self._run_inference(
            prepared_input
        )

        result = self.decoder.decode(
            output=output,
            original_width=int(
                metadata["original_width"]
            ),
            original_height=int(
                metadata["original_height"]
            ),
            scale=float(
                metadata["scale"]
            ),
            pad_x=float(
                metadata["pad_x"]
            ),
            pad_y=float(
                metadata["pad_y"]
            ),
        )

        return result

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        interpreter = getattr(
            self,
            "interpreter",
            None
        )

        if interpreter is not None:

            try:
                self.interpreter = None

            except Exception:
                pass

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.close()
        return False

    # ========================================================
    # DESTRUCTOR
    # ========================================================

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass
