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
android.minapi = 24
android.archs = arm64-v8a
android.accept_sdk_license = True

android.permissions = READ_MEDIA_IMAGES
p4a.local_recipes = ./p4a_recipes
[buildozer]

log_level = 2
warn_on_root = 1
