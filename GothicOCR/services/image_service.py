from pathlib import Path

import numpy as np
from PIL import Image


class ImageService:
    """
    تجهيز الصور لنموذج GothicOCR.

    الناتج:
        dtype: float32
        shape: [1, 3, 1024, 1024]

    يستخدم Letterbox للحفاظ على أبعاد الصورة.
    """

    def __init__(self, target_size=1024, fill=114):

        self.target_size = int(target_size)
        self.fill = int(fill)

        if self.target_size <= 0:
            raise ValueError(
                "target_size يجب أن يكون أكبر من صفر."
            )

        if not 0 <= self.fill <= 255:
            raise ValueError(
                "fill يجب أن يكون بين 0 و255."
            )

    def _prepare_rgb(self, image):

        image = np.asarray(
            image,
            dtype=np.uint8
        )

        if image.ndim != 3:
            raise ValueError(
                f"الصورة يجب أن تكون ثلاثية الأبعاد: "
                f"{image.shape}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"الصورة يجب أن تكون RGB: "
                f"{image.shape}"
            )

        original_h, original_w = image.shape[:2]

        if original_w <= 0 or original_h <= 0:
            raise ValueError(
                "أبعاد الصورة غير صحيحة."
            )

        scale = min(
            self.target_size / original_w,
            self.target_size / original_h,
        )

        new_w = max(
            1,
            int(round(original_w * scale))
        )

        new_h = max(
            1,
            int(round(original_h * scale))
        )

        pil_image = Image.fromarray(
            image,
            mode="RGB"
        )

        resized = pil_image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (
                self.target_size,
                self.target_size,
            ),
            (
                self.fill,
                self.fill,
                self.fill,
            ),
        )

        pad_x = (
            self.target_size - new_w
        ) // 2

        pad_y = (
            self.target_size - new_h
        ) // 2

        canvas.paste(
            resized,
            (pad_x, pad_y),
        )

        array = np.asarray(
            canvas,
            dtype=np.float32,
        )

        chw = np.transpose(
            array,
            (2, 0, 1),
        )

        prepared_input = np.expand_dims(
            chw,
            axis=0,
        )

        prepared_input = np.ascontiguousarray(
            prepared_input,
            dtype=np.float32,
        )

        expected_shape = (
            1,
            3,
            self.target_size,
            self.target_size,
        )

        if prepared_input.shape != expected_shape:
            raise RuntimeError(
                f"فشل تجهيز الصورة. "
                f"الناتج: {prepared_input.shape}"
            )

        meta = {
            "original_width": original_w,
            "original_height": original_h,
            "resized_width": new_w,
            "resized_height": new_h,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "input_size": self.target_size,
        }

        return prepared_input, meta

    def load_and_prepare(self, image_path):

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"الصورة غير موجودة: {path}"
            )

        try:

            with Image.open(path) as source:
                image = source.convert("RGB")
                array = np.asarray(
                    image,
                    dtype=np.uint8
                )

        except Exception as exc:

            raise RuntimeError(
                f"تعذر فتح الصورة: {path}"
            ) from exc

        return self._prepare_rgb(array)

    def prepare(self, image):

        return self._prepare_rgb(image)