
from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from plyer import filechooser


# ============================================================
# COLORS
# ============================================================

BG = (0.99, 0.95, 0.98, 1)
PINK = (0.78, 0.32, 0.65, 1)
DARK = (0.18, 0.12, 0.17, 1)
WHITE = (1, 1, 1, 1)


# ============================================================
# PINK BUTTON
# ============================================================

class PinkButton(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)

        self.color = WHITE
        self.font_size = "16sp"
        self.bold = True

        with self.canvas.before:
            Color(*PINK)
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[24]
            )

        self.bind(
            pos=self.update_rect,
            size=self.update_rect
        )

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================
# OUTLINE BUTTON
# ============================================================

class OutlineButton(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)

        self.color = PINK
        self.font_size = "16sp"
        self.bold = True

        with self.canvas.before:
            Color(*PINK)
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[24]
            )

        self.bind(
            pos=self.update_rect,
            size=self.update_rect
        )

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================
# MAIN SCREEN
# ============================================================

class GothicOCRScreen(BoxLayout):

    def __init__(self, **kwargs):

        super().__init__(
            orientation="vertical",
            padding=[30, 25, 30, 25],
            spacing=14,
            **kwargs
        )

        # ----------------------------------------------------
        # BACKGROUND
        # ----------------------------------------------------

        with self.canvas.before:
            Color(*BG)

            self.bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size
            )

        self.bind(
            pos=self.update_background,
            size=self.update_background
        )


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        self.add_widget(
            Label(
                text="GOTHIC OCR",
                color=DARK,
                font_size="29sp",
                bold=True,
                size_hint_y=None,
                height=48
            )
        )


        # ----------------------------------------------------
        # SUBTITLE
        # ----------------------------------------------------

        self.add_widget(
            Label(
                text="Gothic script recognition",
                color=DARK,
                font_size="14sp",
                size_hint_y=None,
                height=28
            )
        )


        # ----------------------------------------------------
        # GALLERY / CAMERA
        # ----------------------------------------------------

        top_buttons = BoxLayout(
            orientation="horizontal",
            spacing=12,
            size_hint_y=None,
            height=55
        )


        self.gallery_button = OutlineButton(
            text="Gallery"
        )

        self.gallery_button.bind(
            on_release=self.open_gallery
        )


        self.camera_button = PinkButton(
            text="Camera"
        )

        # Camera will be connected later.


        top_buttons.add_widget(
            self.gallery_button
        )

        top_buttons.add_widget(
            self.camera_button
        )

        self.add_widget(top_buttons)


        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        image_container = BoxLayout(
            orientation="vertical",
            padding=10
        )


        self.preview = Image(
            source="",
            allow_stretch=True,
            keep_ratio=True
        )


        image_container.add_widget(
            self.preview
        )

        self.add_widget(
            image_container
        )


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        self.analyze_button = PinkButton(
            text="Analyze",
            size_hint_y=None,
            height=58
        )

        self.analyze_button.bind(
            on_release=self.analyze
        )

        self.add_widget(
            self.analyze_button
        )


        # ----------------------------------------------------
        # EXTRACTED TEXT
        # ----------------------------------------------------

        self.add_widget(
            Label(
                text="Extracted Text",
                color=DARK,
                font_size="18sp",
                bold=True,
                size_hint_y=None,
                height=35
            )
        )


        self.result = Label(
            text="—",
            color=DARK,
            font_size="25sp",
            halign="center",
            valign="middle",
            size_hint_y=None,
            height=65
        )


        self.result.bind(
            size=lambda instance, value:
            setattr(instance, "text_size", value)
        )


        self.add_widget(
            self.result
        )


        # ----------------------------------------------------
        # SAVE / COPY
        # ----------------------------------------------------

        bottom_buttons = BoxLayout(
            orientation="horizontal",
            spacing=12,
            size_hint_y=None,
            height=55
        )


        self.save_button = OutlineButton(
            text="Save TXT"
        )


        self.copy_button = PinkButton(
            text="Copy Text"
        )


        bottom_buttons.add_widget(
            self.save_button
        )

        bottom_buttons.add_widget(
            self.copy_button
        )


        self.add_widget(
            bottom_buttons
        )


        # ----------------------------------------------------
        # ANALYZE ANOTHER
        # ----------------------------------------------------

        self.another_button = OutlineButton(
            text="Analyze Another Image",
            size_hint_y=None,
            height=55
        )


        self.another_button.bind(
            on_release=self.reset_image
        )


        self.add_widget(
            self.another_button
        )


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.selected_image = None


    # ========================================================
    # BACKGROUND
    # ========================================================

    def update_background(self, *args):

        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


    # ========================================================
    # OPEN GALLERY
    # ========================================================

    def open_gallery(self, *args):

        print("Opening Gallery...")

        try:

            filechooser.open_file(
                on_selection=self.gallery_selected,
                filters=[
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.webp"
                ],
                multiple=False
            )

        except Exception as error:

            print("Gallery error:", error)

            self.result.text = "Gallery unavailable"


    # ========================================================
    # IMAGE SELECTED
    # ========================================================

    def gallery_selected(self, selection):

        if not selection:
            print("No image selected.")
            return


        image_path = selection[0]

        print("=" * 50)
        print("IMAGE SELECTED")
        print(image_path)
        print("=" * 50)


        self.selected_image = image_path

        self.preview.source = image_path
        self.preview.reload()

        self.result.text = "Image selected"


    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(self, *args):

        if not self.selected_image:

            self.result.text = "Select an image first."

            print("No image selected.")

            return


        print("Image ready for OCR:")
        print(self.selected_image)


        self.result.text = "Ready for OCR"


    # ========================================================
    # RESET
    # ========================================================

    def reset_image(self, *args):

        self.selected_image = None

        self.preview.source = ""
        self.preview.reload()

        self.result.text = "—"

        print("Image cleared.")


# ============================================================
# APP
# ============================================================

class GothicOCRApp(App):

    def build(self):

        Window.clearcolor = BG

        return GothicOCRScreen()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    GothicOCRApp().run()
