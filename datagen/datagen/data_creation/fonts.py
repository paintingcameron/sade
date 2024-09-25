from PIL import ImageFont
import random
import os

class Fonts:
    _fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font-families")
    
    FONTS = {
        "UBUNTU_MONO": "UbuntuMono-Regular.ttf",
    }

    DEFAULT = ImageFont.truetype(os.path.join(_fonts_path, FONTS["UBUNTU_MONO"]), 64)

    @staticmethod
    def getRandomFont(size=64):
        font = random.choice(list(Fonts.FONTS.values()))
        return Fonts.getFont(font, size)

    @staticmethod
    def getFont(font, size=64):
        return ImageFont.truetype(os.path.join(Fonts._fonts_path, font), size)

def get_font_str(font):
    for name, path in Fonts.FONTS.items():
        if font == path:
            return name
    raise KeyError("Unknown font")