
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
            size_hint_y: None
            height: dp(170)

            Button:
                text: "اختيار صورة"
                on_release: root.select_image()

            Button:
                text: "التعرّف على النص"
                on_release: root.run_ocr()

            Button:
                text: "مسح"
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
"""


Builder.load_string(KV)


class MainScreen(Screen):

    status_text = StringProperty(
        "اختر صورة للبدء"
    )

    result_text = StringProperty("")

    selected_image = StringProperty("")

    model_service = None

    # --------------------------------------------------------
    # MODEL SERVICE
    # --------------------------------------------------------

    def set_model_service(self, model_service):
        self.model_service = model_service

    # --------------------------------------------------------
    # SELECT IMAGE
    # --------------------------------------------------------

    def select_image(self):

        app = self.manager.app if self.manager else None

        if app is None:
            from kivy.app import App
            app = App.get_running_app()

        if app is not None and hasattr(
            app,
            "open_file_chooser"
        ):
            app.open_file_chooser()
        else:
            self.status_text = (
                "تعذر فتح اختيار الصورة"
            )

    # --------------------------------------------------------
    # SET IMAGE
    # --------------------------------------------------------

    def set_image(self, image_path):

        self.selected_image = str(
            image_path
        )

        self.status_text = (
            "تم اختيار الصورة"
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    def run_ocr(self):

        if not self.selected_image:
            self.status_text = (
                "اختر صورة أولًا"
            )
            return

        if self.model_service is None:
            self.status_text = (
                "النموذج غير متصل"
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

            print(
                "OCR error:",
                error
            )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear_result(self):

        self.selected_image = ""
        self.result_text = ""

        self.status_text = (
            "اختر صورة للبدء"
        )
