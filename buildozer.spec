[app]

title = GothicOCR
package.name = gothicocr
package.domain = org.gothicocr

source.dir = .
source.include_exts = py,png,jpg,jpeg,webp,ttf,tflite,json

version = 1.0.0

requirements = python3,kivy,pillow,numpy,tflite-runtime

orientation = portrait
fullscreen = 0

android.api = 35
android.minapi = 23
android.archs = arm64-v8a
android.accept_sdk_license = True

android.permissions = READ_MEDIA_IMAGES

[buildozer]

log_level = 2
warn_on_root = 1
