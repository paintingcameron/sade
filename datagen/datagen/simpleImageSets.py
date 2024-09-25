import random
import string

from abc import ABC, abstractmethod

from PIL import ImageFont, Image
import cv2

import datagen.data_creation.augmentations as augs
from datagen.data_creation.utils import get_contrasting_color, get_textsize
from datagen.data_creation.fonts import Fonts

LETTERS = string.ascii_letters + string.digits
PADDING = 20
TRANSPARENT_COLOR = (0,0,0,0)
BLACK = (0,0,0,255)
WHITE = (255,255,255,255)

class TextImage(ABC):
    @classmethod  
    @abstractmethod  
    def create_random(text, font, *args, **kwargs):
        pass

    @classmethod  
    @abstractmethod  
    def create(*args, **kwargs):
        pass

def random_image(length, creater: TextImage=None, image_sets=None):
    ALL_SETS = [
        SoftBlur,
        Fenced,
        LightReflected,
        Enlarged,
        Shadow
    ]

    if creater == None:
        creater = random.choice()

    text = "".join(random.choice(LETTERS) for _ in range(length))
    font = Fonts.getRandomFont()

    return creater.create_random(text, font)

def center_crop(image, crop_width = 280, crop_height=100):

    width, height = image.size

    box = (
        (width - crop_width)//2,    # Left
        (height - crop_height)//2,  # top
        (width + crop_width)//2,    # Right
        (height + crop_height)//2   # bottom
    )

    return image.crop(box)

def random_contrasting_colors():
    text_color = tuple([random.randint(0,255) for i in range(3)] + [255])
    background_color = get_contrasting_color(text_color)

    return text_color, background_color

class BasicText(TextImage):
    def create_random(text, font, text_color=None, background_color=None):

        if text_color == None:
            text_color = [random.randint(0, 255) for _ in range(3)] + [255]
        if background_color == None:
            background_color = (0,0,0,0)

        scale_factor = 0.8
        max_rotation = 10

        return BasicText.create(text, font, text_color, background_color,
                                scale_factor, max_rotation)

    def create(text, font, text_color, background_color, scale_factor=0, 
               max_rotation=0, padding=PADDING):
        
        image = augs.AugmentationSequence(*[
            augs.TextImage(text, font, background_color, text_color, padding=padding),
            augs.DownScale(scale_factor, fill_color=background_color),
            augs.RotateImage(max_rotation, background_color)
        ])()

        return image

class SoftBlur(TextImage):
    def create_random(text, font, text_color=None, background_color=None):
        tc, bc = random_contrasting_colors()
        if text_color is None:
            text_color = tc
        if background_color is None:
            background_color = bc

        blur_intensity = random.uniform(1,3)
        noise_intensity = random.randint(50, 100)

        return SoftBlur.create(text, font, text_color, background_color, 
                               blur_intensity, noise_intensity)

    def create(text, font, text_color, background_color, blur_intensity,
               noise_intensity, padding=PADDING):
        image = BasicText.create_random(text, font, text_color)

        image = augs.ApplyBackground(augs.AugmentationSequence(*[
                augs.BlankImage(text, font, background_color=background_color, padding=padding),
                augs.StaticNoise(noise_intensity),
        ]))(image)

        image = augs.Blur(augs.Blur.GAUSSIAN_BLUR, blur_intensity)(image)

        return image
    
class Fenced(TextImage):

    def create_random(text, font, text_color=None, background_color=None):
        tc, bc = random_contrasting_colors()
        if text_color is None:
            text_color = tc
        if background_color is None:
            background_color = bc

        fence_angle = random.randint(-45, 45)
        fence_width = random.randint(3, 5)
        between_width = random.randint(25, 30)

        return Fenced.create(text, font, text_color, background_color,
                             fence_angle, fence_width, between_width)
        

    def create(text, font, text_color, background_color, fence_angle, fence_width, between_width):

        fence_color = (71, 50, 26, 255)

        image = BasicText.create_random(text, font, text_color, background_color)
        image = augs.AugmentationSequence(*[
            augs.RepeatLine(augs.Line(
                    line_color=fence_color,
                    line_loc=(0,0),
                    line_width=fence_width,
                    angle=fence_angle
                ),
                between_width
            ),
            augs.Blur(augs.Blur.GAUSSIAN_BLUR, 0.8),
        ])(image)

        return image
    
class LightReflected(TextImage):
    def create_random(text, font, text_color=None, background_color=None):
        tc, bc = random_contrasting_colors()
        if text_color is None:
            text_color = tc
        if background_color is None:
            background_color = bc
        
        width, height = get_textsize(font, text)

        light_intensity = random.uniform(250, 300)
        light_radius = random.randint(100, 150)
        light_var = random.randint(200, 250)
        num_lights = random.randint(1, 4)

        light_locations = [(random.randint(0, width), random.randint(0, height)) 
                           for _ in range(num_lights)]

        return LightReflected.create(text, font, text_color, background_color, 
                                     light_locations, light_intensity, light_radius,
                                     light_var)

    def create(text, font, text_color, background_color, light_locations, 
               light_intensity, light_radius, light_var):
        
        light_covar = ((light_var, 0), (0, light_var))

        image = BasicText.create_random(text, font, text_color, background_color)

        image = augs.AugmentationSequence(*[
            augs.LightReflection(radius=light_radius, center=loc, 
                                 intensity=light_intensity, cov=light_covar) 
                                 for loc in light_locations
            ]
        )(image)

        return image
    
class Enlarged(TextImage):
    
    def create_random(text, font, text_color=None, background_color=None, max_scale=2):
        tc, bc = random_contrasting_colors()
        if text_color is None:
            text_color = tc
        if background_color is None:
            background_color = bc

        font_size = random.randint(20, 30)
        small_font = ImageFont.FreeTypeFont(font.path, size=font_size)

        final_size = get_textsize(font, text)
        final_size = (final_size[0] + max_scale*PADDING, final_size[1] + max_scale*PADDING)

        return Enlarged.create(text, small_font, text_color, background_color, final_size)

    def create(text, font, text_color, background_color, final_size):

        image = BasicText.create_random(text, font, text_color, background_color)

        image = image.resize(final_size, resample=Image.NEAREST)

        return image

class Shadow(TextImage):

    def create_random(text, font, text_color=None, background_color=None):
        tc, bc = random_contrasting_colors()
        if text_color is None:
            text_color = tc
        if background_color is None:
            background_color = bc

        max_offset = 4
        offset = (random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset))

        shadow_intensity = random.randint(40, 60)
        shadow_color = tuple([text_color[i] - shadow_intensity for i in range(3)] + [255])

        scale_factor = 0.7
        max_rotation = 10

        return Shadow.create(text, font, text_color, background_color, offset, shadow_color,
                             scale_factor, max_rotation)


    def create(text, font, text_color, background_color, shadow_offset, shadow_color, scale_factor,
               max_rotation, padding=PADDING):

        image = augs.AugmentationSequence(*[
            augs.TextImage(text, font, (0,0,0,0), text_color, padding=padding),
            augs.ApplyBackground(augs.AugmentationSequence(*[
                augs.TextImage(text, font, background_color, shadow_color, padding=PADDING),
            ]), shadow_offset),
            augs.DownScale(scale_factor, fill_color=background_color),
            augs.RotateImage(max_rotation, background_color)
        ])()

        return image

class VenezualaNumberPlate(TextImage):
    def create_random(text, font):
        return VenezualaNumberPlate.create(text, font)

    def create(text, font, background_color):
        
        above_text = "".join([random.choice(' ' + string.ascii_uppercase) for _ in range(20)])
        below_text = "".join([random.choice(string.ascii_uppercase) for _ in range(8)])
        small_text_font = Fonts.getFont(Fonts.FONTS['UBUNTU_MONO'], 18)

        hole_points = (70, 190, 40, 220)
        dark_hole = (61, 60, 60, 255)
        light_hole = (217, 217, 206)

        background_augs = augs.AugmentationSequence(*[
            augs.BlankImage(text, font, (200, 200, 200, 255), 10),
            augs.Resize(256, 256),
            augs.Line((194, 22, 10,255), (128, 229), 70, (1,0.1)),
            augs.Line((7, 147, 222, 255), (128, 160), 70, (1,0.1)),
            augs.Line((249, 255, 135, 255), (128, 100), 70, (1,0.1)),
            augs.DrawText(below_text, small_text_font, offset=(0, 90)),
            augs.DrawText(above_text, small_text_font, offset=(0, -75)),
            augs.Hole((hole_points[0], hole_points[2]), 11, dark_hole),
            augs.Hole((hole_points[0], hole_points[2]), 6, light_hole),
            augs.Hole((hole_points[1], hole_points[2]), 11, dark_hole),
            augs.Hole((hole_points[1], hole_points[2]), 6, light_hole),
            augs.Hole((hole_points[0], hole_points[3]), 7, dark_hole),
            augs.Hole((hole_points[1], hole_points[3]), 7, dark_hole),
            augs.Line(dark_hole, (128, 5), 12),
            augs.Line(light_hole, (128, 5), 3),
            augs.Line(dark_hole, (128, 251), 12),
            augs.Line(light_hole, (128, 251), 3),
            augs.Line(dark_hole, (5, 128), 12, (0,1)),
            augs.Line(light_hole, (5, 128), 3, (0,1)),
            augs.Line(dark_hole, (251, 128), 12, (0,1)),
            augs.Line(light_hole, (251, 128), 3, (0,1)),
        ])

        img = augs.AugmentationSequence(*[
            augs.BlankImage(text, font, (0,0,0,0), (10)),
            augs.DrawText(text, font, offset=(0, -5)),
            augs.Resize(250, 280),
            augs.ApplyBackground(background_aug=background_augs),
            augs.StaticNoise(100),
            augs.RotateImage(10, fill_color=background_color),
            augs.Resize(64, 64, cv2.INTER_LINEAR),
            augs.Resize(256, 256, cv2.INTER_NEAREST),
        ])()

        return img