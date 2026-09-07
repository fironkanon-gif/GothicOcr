
# ============================================================
# GothicOCR - Main
# ============================================================

from kivy.app import App

from screens.main_screen import MainScreen
from services.model_service import ModelService


class GothicOCRApp(App):

    title = "GothicOCR"

    def build(self):

        screen = MainScreen()

        try:

            model_service = ModelService()

            screen.set_model_service(
                model_service
            )

            screen.status_text = (
                "النموذج جاهز — اختر صورة"
            )

        except Exception as error:

            print(
                "Model initialization error:",
                error
            )

            screen.status_text = (
                "تعذر تحميل النموذج"
            )

        return screen


if __name__ == "__main__":
    GothicOCRApp().run()
