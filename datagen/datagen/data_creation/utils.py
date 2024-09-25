from math import sqrt, sin, cos, radians, atan2, degrees
from typing import Tuple
from warnings import warn
from random import randint
from PIL import Image, ImageDraw
import numpy as np


def clip(c):
    if c < 0:
        return 0
    elif c > 255:
        return 255
    else:
        return round(c)
    
def intensify_color(image, p, i):
    pixel_color = image.getpixel(p)

    return tuple([clip(c + 255 * i) for c in pixel_color[:3]] + [pixel_color[3]])

def valid_point(image, p):
    return 0 <= p[0] < image.size[0] and 0 <= p[1] < image.size[1]

def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def _relative_lumance(color):
            get_l = lambda x: x/12.92 if x <= 0.03928 else ((x + 0.055)/1.055)**2.4

            return 0.2126 * get_l(color[0]) + \
                0.7152 * get_l(color[1]) + \
                0.0722 * get_l(color[2])

def get_contrasting_color(c1:Tuple[int, int,int]):
    c1_ave = sum(c1[:3])/3
    c1_L = _relative_lumance(c1)

    fails = -1
    cr = 0
    while cr < 4.5:
        c2 = tuple([randint(0,255) for i in range(3)] + [255])
        c2_L = _relative_lumance(c2)

        ave_text_color = sum([c for c in c2[:3]])/3

        if c1_ave > ave_text_color:
            lighter = c1_L
            darker = c2_L
        else:
            lighter = c2_L
            darker = c1_L

        cr = (lighter + 0.5) / (darker + 0.05)

        fails += 1

        if fails == 30:
            warn("Image generated with suboptimal contrast")
            break

    return c2

def get_line_points(p1, angle, length, width):

    cos_angle = cos(radians(angle))
    sin_angle = sin(radians(angle))

    get_x = lambda l: round(p1[0] + cos_angle * l)
    get_y = lambda l: round(p1[1] + sin_angle * l)
     
    points = [(get_x(l), get_y(l)) for l in range(length)]

    return points


def get_image_set(set_name):

    return getattr(datagen.simpleImageSets, set_name)


def get_textsize(font, text):
    left, top, right, bottom = font.getbbox(text)
    width = right - left
    height = bottom - top
    return width, height


def translate_image(image, offset_x, offset_y):
    width, height = image.size
    translation_matrix = (1, 0, offset_x, 0, 1, offset_y)
    translated_image = image.transform((width, height), Image.AFFINE, translation_matrix)
    return translated_image


def vector_to_angle(vector):
    x, y = vector
    angle_radians = atan2(y, x)
    angle_degrees = degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees += 360
    return angle_degrees