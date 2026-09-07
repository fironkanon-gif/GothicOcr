# ============================================================
# GothicOCR - Main Application
# ============================================================

from pathlib import Path

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.button import Button

from screens.main_screen import MainScreen
from services.model_service import ModelService


KV = """
<RootWidget>:
"""


class RootWidget(MainScreen):
    pass


Builder.load_string(KV)


class GothicOCRApp(App):

    title = "GothicOCR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_service = None
        self.main_screen = None

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    def build(self):

        self.main_screen = RootWidget()

        # Connect the model service.
        try:
            self.model_service = ModelService()

            self.main_screen.set_model_service(
                self.model_service
            )

            self.main_screen.status_text = (
                "النموذج جاهز — اختر صورة"
            )

        except Exception as error:

            self.main_screen.status_text = (
                "تعذر تحميل النموذج"
            )

            print(
                "Model initialization error:",
                error
            )

        return self.main_screen

    # --------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------

    def on_start(self):

        print("=" * 60)
        print("GothicOCR started")
        print("=" * 60)

        if self.model_service is not None:

            try:
                print(
                    self.model_service.get_model_info()
                )

            except Exception as error:
                print(
                    "Model information error:",
                    error
                )

    # --------------------------------------------------------
    # FILE SELECTION
    # --------------------------------------------------------

    def open_file_chooser(self):

        chooser = FileChooserListView(
            path=str(
                Path.home()
            ),
            filters=[
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.webp",
            ],
        )

        select_button = Button(
            text="اختيار",
            size_hint_y=None,
            height="50dp",
        )

        content = __import__(
            "kivy.uix.boxlayout",
            fromlist=["BoxLayout"]
        ).BoxLayout(
            orientation="vertical",
        )

        content.add_widget(chooser)
        content.add_widget(select_button)

        popup = Popup(
            title="اختيار صورة",
            content=content,
            size_hint=(0.95, 0.95),
        )

        def select_file(instance):

            if chooser.selection:

                image_path = chooser.selection[0]

                self.main_screen.set_image(
                    image_path
                )

                popup.dismiss()

        select_button.bind(
            on_release=select_file
        )

        popup.open()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def on_stop(self):

        print("GothicOCR stopped")


if __name__ == "__main__":
    GothicOCRApp().run()
