============================================================

GOTHIC OCR — MODEL SERVICE

Android / PyJNIus / TensorFlow Lite

============================================================

from pathlib import Path

import numpy as np

from services.image_service import ImageService
from services.text_decoder import TextDecoder

class GothicOCR:
"""
Android OCR service.

Flow:

    image
      ↓
    ImageService
      ↓
    NCHW float32
      ↓
    Direct ByteBuffer
      ↓
    TensorFlow Lite Java Interpreter
      ↓
    Direct Output ByteBuffer
      ↓
    NumPy float32
      ↓
    TextDecoder
"""

INPUT_SHAPE = (1, 3, 1024, 1024)
OUTPUT_SHAPE = (1, 29, 21504)

INPUT_SIZE = 1024

# ========================================================
# INITIALIZATION
# ========================================================

def __init__(self, model_path):

    self.model_path = Path(model_path)

    if not self.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {self.model_path}"
        )

    # ----------------------------------------------------
    # Android / Java imports
    # ----------------------------------------------------

    try:
        from jnius import autoclass
    except Exception as error:
        raise RuntimeError(
            "PyJNIus is required to run GothicOCR on Android."
        ) from error

    try:
        JavaFile = autoclass(
            "java.io.File"
        )

        ByteBuffer = autoclass(
            "java.nio.ByteBuffer"
        )

        ByteOrder = autoclass(
            "java.nio.ByteOrder"
        )

        Interpreter = autoclass(
            "org.tensorflow.lite.Interpreter"
        )

    except Exception as error:
        raise RuntimeError(
            "Failed to load Android TensorFlow Lite classes."
        ) from error

    self._ByteBuffer = ByteBuffer
    self._ByteOrder = ByteOrder

    # ----------------------------------------------------
    # Load TFLite model
    # ----------------------------------------------------

    try:

        java_file = JavaFile(
            str(self.model_path)
        )

        if not java_file.exists():
            raise FileNotFoundError(
                "Java could not find model: "
                f"{self.model_path}"
            )

        self.interpreter = Interpreter(
            java_file
        )

    except Exception as error:

        self.interpreter = None

        raise RuntimeError(
            "Failed to create TensorFlow Lite "
            f"Interpreter: {error}"
        ) from error

    # ----------------------------------------------------
    # Inspect tensors
    # ----------------------------------------------------

    try:

        self.input_details = (
            self.interpreter.getInputTensor(0)
        )

        self.output_details = (
            self.interpreter.getOutputTensor(0)
        )

        input_shape = tuple(
            int(value)
            for value in self.input_details.shape()
        )

        output_shape = tuple(
            int(value)
            for value in self.output_details.shape()
        )

        input_type = str(
            self.input_details.dataType()
        )

        output_type = str(
            self.output_details.dataType()
        )

    except Exception as error:

        self.close()

        raise RuntimeError(
            "Failed to inspect TensorFlow Lite "
            f"tensors: {error}"
        ) from error

    # ====================================================
    # STRICT MODEL VALIDATION
    # ====================================================

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

    # ----------------------------------------------------
    # This implementation requires FLOAT32 tensors.
    # ----------------------------------------------------

    if "FLOAT32" not in input_type.upper():

        self.close()

        raise ValueError(
            "Unsupported input tensor type: "
            f"{input_type}. "
            "Expected FLOAT32."
        )

    if "FLOAT32" not in output_type.upper():

        self.close()

        raise ValueError(
            "Unsupported output tensor type: "
            f"{output_type}. "
            "Expected FLOAT32."
        )

    # ====================================================
    # SERVICES
    # ====================================================

    self.image_service = ImageService(
        target_size=self.INPUT_SIZE
    )

    self.decoder = TextDecoder()

    self.input_shape = input_shape
    self.output_shape = output_shape

    # ----------------------------------------------------
    # Byte sizes
    # ----------------------------------------------------

    self.input_elements = int(
        np.prod(
            self.INPUT_SHAPE
        )
    )

    self.output_elements = int(
        np.prod(
            self.OUTPUT_SHAPE
        )
    )

    self.input_bytes = (
        self.input_elements
        *
        np.dtype(
            np.float32
        ).itemsize
    )

    self.output_bytes = (
        self.output_elements
        *
        np.dtype(
            np.float32
        ).itemsize
    )

    # ====================================================
    # LOG
    # ====================================================

    print(
        "========================================"
    )

    print(
        "GOTHIC OCR MODEL SERVICE"
    )

    print(
        "========================================"
    )

    print(
        "Model:",
        self.model_path
    )

    print(
        "Input:",
        self.input_shape
    )

    print(
        "Output:",
        self.output_shape
    )

    print(
        "Input type:",
        input_type
    )

    print(
        "Output type:",
        output_type
    )

    print(
        "Input bytes:",
        self.input_bytes
    )

    print(
        "Output bytes:",
        self.output_bytes
    )

    print(
        "========================================"
    )

# ========================================================
# PREPARE IMAGE
# ========================================================

def _prepare_image(self, image):
    """
    Prepare either an image path or a NumPy image.

    Returns:

        prepared_input
        metadata
    """

    # ----------------------------------------------------
    # Image path
    # ----------------------------------------------------

    if isinstance(
        image,
        (str, Path)
    ):

        (
            prepared_input,
            metadata,
        ) = self.image_service.load_and_prepare(
            image
        )

    # ----------------------------------------------------
    # NumPy image
    # ----------------------------------------------------

    else:

        array = np.asarray(
            image
        )

        if array.ndim != 3:

            raise ValueError(
                "Image must have shape "
                "[H, W, C]. "
                f"Got {array.shape}."
            )

        if array.shape[2] != 3:

            raise ValueError(
                "Image must have 3 RGB "
                "channels. "
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
            ).astype(
                np.uint8
            )

        (
            prepared_input,
            metadata,
        ) = self.image_service.prepare(
            array
        )

    # ----------------------------------------------------
    # Force float32
    # ----------------------------------------------------

    prepared_input = np.asarray(
        prepared_input,
        dtype=np.float32
    )

    # ====================================================
    # STRICT INPUT VALIDATION
    # ====================================================

    if prepared_input.shape != self.INPUT_SHAPE:

        raise ValueError(
            "Prepared input has wrong shape: "
            f"{prepared_input.shape}. "
            f"Expected {self.INPUT_SHAPE}."
        )

    if not prepared_input.flags[
        "C_CONTIGUOUS"
    ]:

        prepared_input = np.ascontiguousarray(
            prepared_input,
            dtype=np.float32
        )

    if not np.all(
        np.isfinite(
            prepared_input
        )
    ):

        raise ValueError(
            "Prepared input contains "
            "NaN or Inf."
        )

    # ----------------------------------------------------
    # Validate metadata
    # ----------------------------------------------------

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

    if float(
        metadata["scale"]
    ) <= 0:

        raise ValueError(
            "ImageService returned an invalid "
            f"scale: {metadata['scale']}"
        )

    return (
        prepared_input,
        metadata
    )

# ========================================================
# CREATE DIRECT BYTEBUFFER
# ========================================================

def _create_direct_buffer(
    self,
    size_bytes
):
    """
    Create a native-order direct ByteBuffer.
    """

    if size_bytes <= 0:

        raise ValueError(
            f"Invalid ByteBuffer size: "
            f"{size_bytes}"
        )

    buffer = (
        self._ByteBuffer
        .allocateDirect(
            int(size_bytes)
        )
    )

    buffer.order(
        self._ByteOrder.nativeOrder()
    )

    buffer.rewind()

    return buffer

# ========================================================
# WRITE INPUT BUFFER
# ========================================================

def _write_input_buffer(
    self,
    tensor
):
    """
    Convert NumPy float32 tensor
    to Java Direct ByteBuffer.
    """

    tensor = np.asarray(
        tensor,
        dtype=np.float32
    )

    if tensor.shape != self.INPUT_SHAPE:

        raise ValueError(
            "Input tensor shape mismatch: "
            f"{tensor.shape}. "
            f"Expected {self.INPUT_SHAPE}."
        )

    if not np.all(
        np.isfinite(tensor)
    ):

        raise ValueError(
            "Input tensor contains "
            "NaN or Inf."
        )

    tensor = np.ascontiguousarray(
        tensor,
        dtype=np.float32
    )

    raw_bytes = tensor.tobytes(
        order="C"
    )

    if len(raw_bytes) != self.input_bytes:

        raise ValueError(
            "Input byte size mismatch: "
            f"{len(raw_bytes)} != "
            f"{self.input_bytes}"
        )

    input_buffer = (
        self._create_direct_buffer(
            self.input_bytes
        )
    )

    try:

        input_buffer.put(
            bytearray(
                raw_bytes
            )
        )

        input_buffer.rewind()

    except Exception as error:

        raise RuntimeError(
            "Failed to write input "
            f"ByteBuffer: {error}"
        ) from error

    return input_buffer

# ========================================================
# READ OUTPUT BUFFER
# ========================================================

def _read_output_buffer(
    self,
    output_buffer
):
    """
    Read TensorFlow Lite output
    Direct ByteBuffer into NumPy float32.
    """

    try:

        output_buffer.rewind()

        raw_output = bytearray(
            self.output_bytes
        )

        output_buffer.get(
            raw_output
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to read output "
            f"ByteBuffer: {error}"
        ) from error

    output = np.frombuffer(
        raw_output,
        dtype=np.float32
    ).copy()

    if output.size != (
        self.output_elements
    ):

        raise ValueError(
            "Unexpected output element "
            f"count: {output.size}. "
            f"Expected: {self.output_elements}."
        )

    output = output.reshape(
        self.OUTPUT_SHAPE
    )

    if not np.all(
        np.isfinite(output)
    ):

        raise ValueError(
            "TensorFlow Lite output "
            "contains NaN or Inf."
        )

    return output

# ========================================================
# RUN TFLITE
# ========================================================

def _run_inference(
    self,
    prepared_input
):
    """
    Execute TensorFlow Lite using
    Direct ByteBuffers.
    """

    input_buffer = (
        self._write_input_buffer(
            prepared_input
        )
    )

    output_buffer = (
        self._create_direct_buffer(
            self.output_bytes
        )
    )

    try:

        input_buffer.rewind()
        output_buffer.rewind()

        # ------------------------------------------------
        # REAL TFLITE INFERENCE
        # ------------------------------------------------

        self.interpreter.run(
            input_buffer,
            output_buffer
        )

        # ------------------------------------------------
        # Read output
        # ------------------------------------------------

        output = (
            self._read_output_buffer(
                output_buffer
            )
        )

    except Exception as error:

        raise RuntimeError(
            "TensorFlow Lite inference "
            f"failed: {error}"
        ) from error

    return output

# ========================================================
# PREDICT
# ========================================================

def predict(
    self,
    image
):
    """
    Run the complete OCR pipeline.

    Example:

        result = ocr.predict(image)

    Returns:

        {
            "text": str,
            "detections": list,
            "lines": list
        }
    """

    (
        prepared_input,
        metadata,
    ) = self._prepare_image(
        image
    )

    output = (
        self._run_inference(
            prepared_input
        )
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
    """
    Release TensorFlow Lite interpreter.
    """

    interpreter = getattr(
        self,
        "interpreter",
        None
    )

    if interpreter is not None:

        try:

            interpreter.close()

        except Exception as error:

            print(
                "Warning: failed to close "
                f"TensorFlow Lite interpreter: "
                f"{error}"
            )

        finally:

            self.interpreter = None

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
