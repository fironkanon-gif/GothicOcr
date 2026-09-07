from pathlib import Path

from kivy.app import App
from kivy.core.text import LabelBase
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.label import Label


ROOT = Path(__file__).resolve().parent

FONT_PATH = (
    ROOT
    / "fonts"
    / "NotoSansGothic-Regular.ttf"
)


class GothicOCRApp(App):

    def build(self):

        self.title = "Gothic OCR"

        self.selected_image = None

        if FONT_PATH.is_file():
            LabelBase.register(
                name="Gothic",
                fn_regular=str(FONT_PATH),
            )

        self.root_box = BoxLayout(
            orientation="vertical",
            padding=12,
            spacing=8,
        )

        self.status = Label(
            text="جاهز لاختيار صورة",
            size_hint_y=None,
            height=45,
        )

        self.preview = Image(
            allow_stretch=True,
            keep_ratio=True,
        )

        self.result = Label(
            text="",
            font_name=(
                "Gothic"
                if FONT_PATH.is_file()
                else "Roboto"
            ),
        )

        self.gallery_button = Button(
            text="🖼 اختيار صورة",
            size_hint_y=None,
            height=52,
        )

        self.gallery_button.bind(
            on_release=self.choose_gallery
        )

        self.camera_button = Button(
            text="📷 التقاط صورة",
            size_hint_y=None,
            height=52,
        )

        self.camera_button.bind(
            on_release=self.capture_camera
        )

        self.analyze_button = Button(
            text="🤖 تحليل الصورة",
            size_hint_y=None,
            height=52,
        )

        self.analyze_button.bind(
            on_release=self.analyze
        )

        self._show_main_layout()

        return self.root_box

    def _show_main_layout(self):

        self.root_box.clear_widgets()

        self.root_box.add_widget(
            self.status
        )

        self.root_box.add_widget(
            self.preview
        )

        self.root_box.add_widget(
            self.gallery_button
        )

        self.root_box.add_widget(
            self.camera_button
        )

        self.root_box.add_widget(
            self.analyze_button
        )

        self.root_box.add_widget(
            self.result
        )

    def choose_gallery(self, *_):

        chooser = FileChooserListView(
            filters=[
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.webp",
            ]
        )

        chooser.bind(
            on_selection=self._selected
        )

        self.root_box.clear_widgets()

        self.root_box.add_widget(
            chooser
        )

        self.status.text = "اختاري صورة..."

    def _selected(
        self,
        chooser,
        selection,
    ):

        if not selection:
            return

        self.selected_image = selection[0]

        self.preview.source = (
            self.selected_image
        )

        self.preview.reload()

        self.status.text = (
            "تم اختيار الصورة بنجاح"
        )

        self._show_main_layout()

    def capture_camera(self, *_):

        self.status.text = (
            "الكاميرا سيتم ربطها "
            "في مرحلة Android Native."
        )

    def analyze(self, *_):

        if not self.selected_image:

            self.status.text = (
                "اختاري صورة أولًا."
            )

            return

        self.status.text = (
            "جاري تجهيز الصورة..."
        )

        try:

            from services.image_service import (
                ImageService,
            )

            image_service = ImageService()

            prepared_input, metadata = (
                image_service.load_and_prepare(
                    self.selected_image
                )
            )

            self.status.text = (
                "تم تجهيز الصورة بنجاح."
            )

            self.result.text = (
                f"Input shape: "
                f"{prepared_input.shape}"
            )

        except Exception as exc:

            self.status.text = (
                f"خطأ: {exc}"
            )


if __name__ == "__main__":

    GothicOCRApp().run()