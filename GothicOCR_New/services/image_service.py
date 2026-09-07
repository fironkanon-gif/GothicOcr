# ============================================================
# GothicOCR - Image Service
# ============================================================

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from utils.constants import (
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_SIZE,
    SUPPORTED_IMAGE_EXTENSIONS,
)


class ImageService:
    """
    Handles image validation, loading, RGB conversion,
    resizing and preparation for the OCR model.
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def validate_image(image_path: Union[str, Path]) -> bool:
        """
        Check whether the supplied path is a supported image.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        return True

    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """
        Return basic information about an image.
        """

        ImageService.validate_image(image_path)

        path = Path(image_path)

        with Image.open(path) as image:
            return {
                "path": str(path),
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            }

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image using Pillow.
        """

        ImageService.validate_image(image_path)

        image = Image.open(image_path)

        # Make an independent copy so the source file is not
        # kept open after this method returns.
        return image.copy()

    # --------------------------------------------------------
    # RGB CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def to_rgb(image: Image.Image) -> Image.Image:
        """
        Convert an image to RGB.
        """

        if image.mode == "RGB":
            return image

        return image.convert("RGB")

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    @staticmethod
    def resize_for_model(image: Image.Image) -> Image.Image:
        """
        Resize image to the exact model input size.

        Model input:
            [1, 1024, 1024, 3]
        """

        image = ImageService.to_rgb(image)

        return image.resize(
            MODEL_INPUT_SIZE,
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # NUMPY CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def to_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert a PIL image to FLOAT32 numpy data.

        Output shape:
            [H, W, 3]

        Output dtype:
            float32

        Pixel range:
            0.0 - 1.0
        """

        image = ImageService.to_rgb(image)

        array = np.asarray(
            image,
            dtype=np.float32
        )

        array /= 255.0

        return array

    # --------------------------------------------------------
    # MODEL PREPARATION
    # --------------------------------------------------------

    @staticmethod
    def prepare_image(
        image: Union[Image.Image, str, Path]
    ) -> np.ndarray:
        """
        Prepare an image for the TFLite OCR model.

        Final output:
            [1, 1024, 1024, 3]

        dtype:
            float32

        values:
            0.0 - 1.0
        """

        if isinstance(image, (str, Path)):
            image = ImageService.load_image(image)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image or an image path."
            )

        image = ImageService.to_rgb(image)

        image = ImageService.resize_for_model(image)

        array = ImageService.to_numpy(image)

        # Add batch dimension.
        array = np.expand_dims(array, axis=0)

        expected_shape = (
            1,
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
            3,
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"Prepared image has shape {array.shape}, "
                f"expected {expected_shape}."
            )

        if array.dtype != np.float32:
            array = array.astype(np.float32)

        return array


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("==============================================")
    print("GothicOCR ImageService")
    print("==============================================")
    print(f"Model input size: {MODEL_INPUT_WIDTH} x {MODEL_INPUT_HEIGHT}")
    print("Expected tensor: [1, 1024, 1024, 3]")
    print("Expected dtype: float32")
    print("==============================================")
