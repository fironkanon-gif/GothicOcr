
from pathlib import Path

from PIL import Image


# ============================================================
# SUPPORTED FORMATS
# ============================================================

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# IMAGE SERVICE
# ============================================================

class ImageService:

    @staticmethod
    def validate_image(image_path):
        """
        Check whether the selected file exists
        and is a supported image.
        """

        if not image_path:
            return False

        path = Path(image_path)

        if not path.exists():
            return False

        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return False

        return True


    # ========================================================
    # GET IMAGE INFORMATION
    # ========================================================

    @staticmethod
    def get_image_info(image_path):
        """
        Return basic information about the image.
        """

        if not ImageService.validate_image(image_path):
            raise ValueError("Invalid image file.")

        with Image.open(image_path) as image:

            return {
                "path": str(image_path),
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format
            }


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    @staticmethod
    def load_image(image_path):
        """
        Load an image safely using PIL.
        """

        if not ImageService.validate_image(image_path):
            raise ValueError("Invalid image file.")

        image = Image.open(image_path)

        # Make sure the image data is loaded
        image.load()

        return image


    # ========================================================
    # CONVERT TO RGB
    # ========================================================

    @staticmethod
    def to_rgb(image):
        """
        Convert image to RGB.
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    @staticmethod
    def prepare_image(image_path):
        """
        Load and prepare the image.

        Model-specific resizing and normalization
        are handled by GothicOCR.
        """

        image = ImageService.load_image(image_path)

        image = ImageService.to_rgb(image)

        return image


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("ImageService loaded successfully.")
