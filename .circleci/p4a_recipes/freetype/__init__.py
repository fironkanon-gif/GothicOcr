from pythonforandroid.recipe import Recipe
from pythonforandroid.logger import shprint
import sh

class FreeTypeRecipe(Recipe):
    version = '2.14.1'
    url = 'https://downloads.sourceforge.net/project/freetype/freetype2/{version}/freetype-{version}.tar.gz'
    built_libraries = {'libfreetype.so': 'objs/.libs'}

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        return env

recipe = FreeTypeRecipe()
