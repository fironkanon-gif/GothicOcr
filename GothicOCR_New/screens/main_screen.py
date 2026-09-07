# ============================================================
# GothicOCR - Main Screen
# ============================================================

from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen


KV = """
<MainScreen>:

    BoxLayout:
        orientation: "vertical"
        padding: dp(20)
        spacing: dp(15)

        canvas.before:
            Color:
                rgba: 0.08, 0.08, 0.08, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            text: "GothicOCR"
            font_size: "30sp"
            bold: True
            size_hint_y: None
            height: dp(55)

        Label:
            text: root.status_text
            font_size: "16sp"
            size_hint_y: None
            height: dp(40)
            text_size: self.width, None
            halign: "center"
            valign: "middle"

        BoxLayout:
            orientation: "vertical"
            spacing: dp(10)

            Button:
                text: "اختيار صورة"
                size_hint_y: None
                height: dp(55)
                on_release: root.select_image()

            Button:
                text: "التعرّف على النص"
                size_hint_y: None
                height: dp(55)
                on_release: root.run_ocr()

            Button:
                text: "مسح"
                size_hint_y: None
                height: dp(50)
                on_release: root.clear_result()

        Label:
            text: "النص القوطي"
            font_size: "20sp"
            size_hint_y: None
            height: dp(40)

        ScrollView:

            TextInput:
                text: root.result_text
                readonly: True
                font_size: "26sp"
                multiline: True
                size_hint_y: None
                height: max(self.minimum_height, dp(180))
                padding: dp(15)
                background_normal: ""
                background_color: 0.15, 0.15, 0.15, 1
                foreground_color: 1, 1, 1, 1
"""


Builder.load_string(KV)


class MainScreen(Screen):
    """
    Main GothicOCR application screen.

    The screen only handles the user interface.
    Model execution is delegated to ModelService by main.py.
    """

    status_text = StringProperty(
        "اختر صورة للبدء"
    )

    result_text = StringProperty("")

    selected_image = StringProperty("")

    model_service = None

    # --------------------------------------------------------
    # CONNECT MODEL SERVICE
    # --------------------------------------------------------

    def set_model_service(self, model_service):
        """
        Connect the screen to ModelService.
        """

        self.model_service = model_service

    # --------------------------------------------------------
    # IMAGE SELECTION
    # --------------------------------------------------------

    def select_image(self):
        """
        Open a file chooser when available.

        The actual Android file-selection integration will be
        connected from main.py.
        """

        self.status_text = (
            "اختيار الصورة متاح من التطبيق"
        )

    # --------------------------------------------------------
    # SET SELECTED IMAGE
    # --------------------------------------------------------

    def set_image(self, image_path):
        """
        Set the currently selected image.
        """

        self.selected_image = str(
            image_path
        )

        self.status_text = (
            "تم اختيار الصورة"
        )

    # --------------------------------------------------------
    # RUN OCR
    # --------------------------------------------------------

    def run_ocr(self):
        """
        Run OCR through the connected ModelService.
        """

        if not self.selected_image:
            self.status_text = (
                "اختر صورة أولًا"
            )
            return

        if self.model_service is None:
            self.status_text = (
                "خدمة النموذج غير متصلة"
            )
            return

        try:
            self.status_text = (
                "جاري التعرّف..."
            )

            result = self.model_service.predict(
                self.selected_image
            )

            self.result_text = result.get(
                "text",
                ""
            )

            if self.result_text:
                self.status_text = (
                    "تم التعرّف على النص"
                )
            else:
                self.status_text = (
                    "لم يتم العثور على نص"
                )

        except Exception as error:

            self.result_text = ""

            self.status_text = (
                f"حدث خطأ: {error}"
            )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear_result(self):
        """
        Clear selected image and OCR result.
        """

        self.selected_image = ""
        self.result_text = ""

        self.status_text = (
            "اختر صورة للبدء"
        )
