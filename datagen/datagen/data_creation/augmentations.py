from abc import ABC, abstractmethod
from PIL import ImageFont, ImageDraw, Image, ImageOps, ImageFilter, ImageChops
from typing import Tuple, Optional
from random import uniform, randint, sample
import numpy as np
import itertools
from warnings import warn
from math import tan, radians, sin, cos, sqrt
from scipy.stats import multivariate_normal
import cv2

from datagen.data_creation.fonts import Fonts
from datagen.data_creation.utils import (
    intensify_color,
    valid_point,
    distance,
    get_contrasting_color,
    get_line_points,
    clip,
    get_textsize,
    vector_to_angle,
    translate_image
)

class Augmentation(ABC):
    def __init__(self):
        self.is_background = False

    @abstractmethod
    def augment(self, image):
        pass

    def __call__(self, image: Image):
        return self.augment(image)
    
class _Background(Augmentation):
    def __init__(self):
        super().__init__()
        self.is_background = True
    
class AugmentationSequence(_Background):

    def __init__(self, *augmentations: Augmentation):
        super().__init__()
        self.augmentations = augmentations

    def __call__(self, image=None):

        start = 0
        if image == None:
            image = self.augmentations[0](None)
            start = 1

        for augmentation in self.augmentations[start:]:
            image = augmentation(image)

        return image
    
    def augment(self, image):
        return self(image)

class BlankImage(_Background):

    def __init__(self, text: str, font: ImageFont.truetype, 
                 background_color: Tuple[int, int, int, int]=(255, 255, 255, 255), 
                 padding: int=0):
        super().__init__()
        
        self.text = text
        self.font = font
        self.background_color = background_color
        self.padding = padding

    def augment(self, image):
        image_size = get_textsize(self.font, self.text)

        image = Image.new('RGBA', image_size, color=self.background_color)

        if self.padding > 0:
            image = ImageOps.expand(image, border=self.padding, fill=self.background_color)

        return image

class DrawText(Augmentation):

    def __init__(self, text: str, font: ImageFont.truetype,
                 text_color=(0, 0, 0, 255), offset:Optional[Tuple[int, int]]=None):
        super().__init__()
        self.text = text
        self.font = font
        self.text_color = text_color
        self.offset = offset

    def augment(self, image:Image):

        image_shape = image.size
        draw = ImageDraw.Draw(image)

        text_width, text_height = get_textsize(self.font, self.text)
        # x = (image_shape[0] - text_width) // 2
        # y = (image_shape[1] - text_height) // 2
        x = image_shape[0] // 2
        y = image_shape[1] // 2

        if self.offset is not None:
            x += self.offset[0]
            y += self.offset[1]

        draw.text((x,y), text=self.text, font=self.font, fill=self.text_color, anchor='mm')

        return image
    
class TextImage(Augmentation):

    def __init__(self, text: str, font:ImageFont=Fonts.DEFAULT, 
                 background_color: Tuple[int, int, int, int]=(255, 255, 255, 255),
                 text_color: Tuple[int, int, int, int]=(0,0,0, 255), padding=0):
        super().__init__()
        
        self.blank_image = BlankImage(text, font, background_color, padding=padding)
        self.draw_text = DrawText(text, font, text_color)

    def augment(self, image: Image):
        image = self.blank_image(None)
        return self.draw_text(image)
    
class RotateImage(Augmentation):

    def __init__(self, max_rotation, fill_color: Tuple[int, int, int, int]=(255,255,255, 255)):
        super().__init__()

        if 0 < max_rotation > 360:
            raise Exception(f"Rotation angle must be between 0 and 360. Got: {max_rotation}")

        self.max_rotation = max_rotation
        self.fill_color = fill_color

    def augment(self, image: Image):
        
        rotate_angle = uniform(-self.max_rotation, self.max_rotation)

        rotated_image = image.rotate(
                                    rotate_angle, 
                                    fillcolor=self.fill_color,
                                    expand=True,
                                    resample=Image.BILINEAR
                                    )

        return rotated_image
    
class DownScale(Augmentation):

    def __init__(self, factor, random_factor=True, fill_color=(255,255,255,255)):
        super().__init__()

        if 0 < factor > 1:
            raise Exception(f"Down scale by a factor between 0 and 1. Got: {factor}")
        
        self.factor = factor
        self.fill_color = fill_color
        self.random_factor = random_factor

    def augment(self, image:Image):

        factor = uniform(self.factor, 1) if self.random_factor else self.factor

        image_shape = image.size
        y_scaled, x_scaled = round(image_shape[0] * factor), round(image_shape[1] * factor)

        if y_scaled <= 0 or x_scaled <= 0:
            raise ValueError(f"Scaled size must be > 0. Got: {x_scaled}, {y_scaled}")

        scaled_image = image.resize((y_scaled, x_scaled))

        background_image = np.zeros((image_shape[0], image_shape[1], 4), dtype=np.uint8)
        background_image[:] = self.fill_color

        delta = ((image_shape[0] - y_scaled)//2, (image_shape[1] - x_scaled)//2)

        background_image[delta[0]:(delta[0]+y_scaled),
                         delta[1]:(delta[1]+x_scaled),:] = np.asarray(scaled_image).transpose(1,0,2)

        return Image.fromarray(background_image.transpose(1,0,2))

class Resize(Augmentation):
    def __init__(self, width: int, height: int, interpolation=cv2.INTER_LINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def augment(self, image):
        return image.resize((self.width, self.height), self.interpolation)

class ApplyBackground(Augmentation):
    def __init__(self, background_aug: Augmentation, center: Optional[Tuple[int,int]]=None):
        super().__init__()

        if not background_aug.is_background:
            raise Exception(f"Can only apply background to a background augmentation. Got: {background_aug}")
        
        self.background_aug = background_aug
        self.center = center

    def augment(self, image):

        background_image = self.background_aug(None)

        blended_image = Image.new("RGBA", background_image.size, (0,0,0,0))

        if self.center == None:
            x_offset = (background_image.width - image.width) // 2
            y_offset = (background_image.height - image.height) // 2
        else:
            y_offset = self.center[0]
            x_offset = self.center[1]

        blended_image.paste(image, (x_offset, y_offset))

        return Image.alpha_composite(background_image, blended_image)

class BackgroundTopBottomGradient(Augmentation):

    def __init__(self, top_intensity:float=0, bottom_intensity:float=0.5):
        super().__init__()

        self.top_intensity = top_intensity
        self.bottom_intensity = bottom_intensity

    def augment(self, image):
        width, height = image.size

        def _calculate_color(x,y):

            r1 = y/height
            r2 = (height - y)/height

            i = (r1 * self.top_intensity + r2 * self.bottom_intensity) / 2
            return intensify_color(image, (x,y), i)

        [image.putpixel((x,y), _calculate_color(x,y)) for x,y in \
                itertools.product(range(width), range(height))]
        
        return image
    
class BackgroundMiddleGradient(Augmentation):

    def __init__(self, top_intensity:float=1, middle_intensity:float=-1, 
                 bottom_intensity:Optional[int]=None, mid_factor:float=0.5):
        super().__init__()

        if 0 > mid_factor > 1:
            raise Exception(f"mid_factor must be a faction between 0 and 1. Got: {mid_factor}")

        if bottom_intensity == None:
            bottom_intensity = top_intensity

        self.top_intensity = top_intensity
        self.middle_intensity = middle_intensity
        self.bottom_intensity = bottom_intensity
        self.mid_factor = mid_factor

    def augment(self, image):
        width, height = image.size

        d1 = round(height * self.mid_factor)
        d2 = height - d1

        def _calculate_color(x,y):
            if y < d1:
                r1 = y/d1
                r2 = 1 - r1
                i1 = self.top_intensity
                i2 = self.middle_intensity
            else:
                r1 = (y-d1)/d2
                r2 = 1 - r1
                i1 = self.middle_intensity
                i2 = self.bottom_intensity

            return intensify_color(image, (x,y), (i1*r2 + i2*r1)/2)
        
        [image.putpixel((x,y), _calculate_color(x,y)) for x,y in
                itertools.product(range(width), range(height))]
        
        return image

class ColorTopBottomGradient(Augmentation):

    def __init__(self, top_color: Tuple[int, int, int], 
                 bottom_color: Tuple[int, int, int]):
        super().__init__()
        self.top_color = top_color
        self.bottom_color = bottom_color

    def augment(self, image):
        width, height = image.size

        def _calculate_color(x,y):
            r1 = y/height
            r2 = 1 - r1

            return tuple([round(self.top_color[i] * r2) 
                          + round(self.bottom_color[i] * r1) for i in range(3)] \
                          + [image.getpixel((x,y))[3]])
        
        [image.putpixel((x,y), _calculate_color(x,y)) for x, y in 
                itertools.product(range(width), range(height))]
        
        return image

class ColorMiddleGradient(Augmentation):

    def __init__(self, top_color: Tuple[int,int,int], middle_color: Tuple[int,int,int],
                 bottom_color: Tuple[int,int,int], mid_factor:int=0.5):
        super().__init__()
        self.top_color = top_color
        self.middle_color = middle_color
        self.bottom_color = bottom_color
        self.mid_factor = mid_factor
    
    def augment(self, image):

        width, height = image.size
        d1 = round(height * self.mid_factor)
        d2 = height - d1

        def _calculate_color(x,y):
            if y < d1:
                r1 = y/d1
                r2 = 1 - r1
                c1 = self.top_color
                c2 = self.middle_color
            else:
                r1 = (y-d1)/d2
                r2 = 1 - r1
                c1 = self.middle_color
                c2 = self.bottom_color

            return tuple(
                [round(c1[i]*r2) + round(c2[i]*r1) for i in range(3)] + [image.getpixel((x,y))[3]])
        
        [image.putpixel((x,y), _calculate_color(x,y)) for x,y 
                in itertools.product(range(width), range(height))]
        
        return image
    
class RandomBackground(_Background):

    def __init__(self, text, font: ImageFont.truetype, padding=0):
        super().__init__()
        self.text = text
        self.font = font
        self.padding = padding

    def augment(self, image):
        self.background_color = tuple([randint(0,255) for i in range(3)] + [255])

        return BlankImage(self.text, self.font, self.background_color, self.padding)(None)
    
class ContrastedText(Augmentation):
    """
    Creates a text image where the text and background have contrasting colors
    """

    def __init__(self, text:str, font: ImageFont.truetype):
        super().__init__()
        self.text = text
        self.font = font

    """
    Equation for relative luminance comes from: 
    https://www.w3.org/TR/WCAG20/#relativeluminancedef
    """
    def augment(self, image):

        background_color = image.getpixel((0,0)) # Assume background has a single color
        
        text_color = get_contrasting_color(background_color)

        return DrawText(self.text, self.font, text_color)(image)
    
class LightReflection(Augmentation):

    def __init__(self, intensity:float=100, radius:int=30, center:Tuple[int,int]=(0,0), cov=((100, 0), (0,100)),
                 dist=None):
        super().__init__()
        
        if dist == None:
            self.dist = multivariate_normal(center, cov)
            self.center = center
        else:
            self.dist = dist
            self.center = self.dist.mean

        self.intensity = intensity
        self.radius = radius

    def augment(self, image):
        width, height = image.size

        pos = [p for p in itertools.product(range((self.center[0]-self.radius), (self.center[0]+self.radius)),
                                            range((self.center[1]-self.radius), (self.center[1]+self.radius)))
                                            if valid_point(image, p)]
        pos = [p for p in pos if distance(self.center, p) < self.radius]
        intensities = self.dist.pdf(pos) * self.intensity

        [image.putpixel(p, intensify_color(image, p, 2*i))
                                for p, i in zip(pos, intensities)]
        
        return image

class Blur(Augmentation):

    STANDARD_BLUR = 0
    BOX_BLUR = 1
    GAUSSIAN_BLUR = 2

    def __init__(self, type:int=STANDARD_BLUR, radius:int=5):
        super().__init__()
        
        if 0 > type > 2:
            raise Exception(f"Blur type not recognized. Got: {type}")
        
        self.type = type
        self.radius = radius

    def augment(self, image):

        if self.type == Blur.STANDARD_BLUR:
            image = image.filter(ImageFilter.BLUR)
        elif self.type == Blur.BOX_BLUR:
            image = image.filter(ImageFilter.BoxBlur(self.radius))
        else: # Guassian blur
            image = image.filter(ImageFilter.GaussianBlur(self.radius))

        return image

class StaticNoise(Augmentation):

    def __init__(self, intensity:float):
        super().__init__()

        self.intensity = intensity

    def augment(self, image):

        np_image = np.asarray(image)
        alpha = np_image[:,:,3]
        np_image = np_image[:,:,:3]

        noise = np.zeros_like(alpha)
        cv2.randu(noise, -self.intensity, self.intensity)
        noise = np.repeat(noise[:,:, np.newaxis], 3, axis=2)

        np_image = cv2.add(np_image, noise)

        np_image = np.dstack((np_image, alpha))

        return Image.fromarray(np_image)
    
class Hole(Augmentation):

    def __init__(self, center:Tuple[int, int], radius:int,
                 hole_color:Tuple[int, int, int, int]):
        super().__init__()

        if len(center) < 2 and center[0] < 0 and center[1] < 1:
            raise Exception(f"Center coordinates invalid. Got: {center}")
        if radius <= 0:
            raise Exception(f"Radius must be a positive number. Got: {radius}")
        
        self.center = center
        self.radius = radius
        self.hole_color = hole_color

    def augment(self, image):

        draw = ImageDraw.Draw(image)

        draw.ellipse((self.center[0]-self.radius, self.center[1]-self.radius,
                      self.center[0]+self.radius, self.center[1]+self.radius), fill=self.hole_color)
        
        return image

class RandomHole(Augmentation):

    def __init__(self, num_holes:int=1, radius:int=5, color=None):
        super().__init__()
        
        if num_holes < 0:
            raise Exception(f"Number of holes must be a positive number. Got: {num_holes}")
        if radius < 0:
            raise Exception(f"Radius must be a postive number. Got: {radius}")
        
        self.num_holes = num_holes
        self.radius = radius
        self.color = color

    def augment(self, image):
        width, height = image.size

        hole_gen = Hole(image.size, self.radius, (0,0,0,0))

        for i in range(self.num_holes):
            center = (randint(0, width-1), randint(0, height-1))
            if self.color == None:
                hole_color  = get_contrasting_color(image.getpixel((center[0], center[1]))[:3])
            else:
                hole_color = self.color

            hole_gen.center = center
            hole_gen.hole_color = hole_color

            image = hole_gen(image)
            
        return image

class Smudge(Augmentation):

    def __init__(self, radius: int = 5, intensity: int = 10):
        self.radius = radius
        self.intensity = intensity

    def augment(self, image):
        width, height = image.size
        smudged_image = image.copy()

        for _ in range(self.intensity):
            # Randomly select a region to smudge
            x1 = randint(0, width - self.radius)
            y1 = randint(0, height - self.radius)
            x2 = x1 + self.radius
            y2 = y1 + self.radius

            # Crop the region, apply blur, and paste it back
            region = smudged_image.crop((x1, y1, x2, y2))
            blurred_region = region.filter(ImageFilter.GaussianBlur(self.radius))
            smudged_image.paste(blurred_region, (x1, y1))

        return smudged_image

class DirectionalSmudges(Augmentation):

    def __init__(self, radius: int = 5, smudges: int = 10, direction: tuple = (1, 0), length: int = 50):
        self.radius = radius
        self.smudges = smudges
        self.direction = direction
        self.length = length

    def augment(self, image):
        width, height = image.size
        smudged_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        for _ in range(self.smudges):
            cx = randint(self.radius, width - self.radius)

            mask = Image.new('RGBA', (width, height), color=(0,0,0,0))
            mask_draw = ImageDraw.Draw(mask)

            mask_draw.ellipse((cx - self.radius, 0, cx + self.radius, height), fill=(1,1,1,1))

            for i in range(self.length, 0, -1):
                offset_x = int(self.direction[0] * i)
                offset_y = int(self.direction[1] * i)
                alpha = int((self.length - i) / self.length * 255)

                image_alpha = np.array(image.split()[3].point(lambda p: alpha if p == 255 else 0))
                region_with_alpha = Image.merge('RGBA', (*image.split()[:3],Image.fromarray(np.array(mask.split()[3]) * image_alpha)))
                shifted_region = translate_image(region_with_alpha, offset_x, offset_y)
                smudged_image.paste(shifted_region, (0,0), shifted_region)

        smudged_image.paste(image, (0, 0), image)

        return smudged_image


class FullSmudge(Augmentation):
    def __init__(self, direction: tuple = (1, 0), length: int = 50):
        dist_mag = sum([abs(d) for d in direction])
        self.direction = tuple([d/dist_mag for d in direction])
        self.length = length

    def augment(self, image):
        width, height = image.size
        smudged_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        mask = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([(0, 0), (width, height)], fill=(1, 1, 1, 1))

        for i in range(self.length, 0, -1):
            offset_x = -int(self.direction[0] * i)
            offset_y = int(self.direction[1] * i)
            alpha = int((self.length - i) / self.length * 255)

            image_alpha = np.array(image.split()[3].point(lambda p: min(alpha, p) if p > 0 else 0))
            region_with_alpha = Image.merge('RGBA', (*image.split()[:3], Image.fromarray(np.array(mask.split()[3]) * image_alpha)))
            shifted_region = translate_image(region_with_alpha, offset_x, offset_y)
            smudged_image.paste(shifted_region, (0, 0), shifted_region)

        # smudged_image.paste(image, (0, 0), image)

        return smudged_image


class Scratch(Augmentation):

    def __init__(self, length:int=10, direction=(1,0), 
                 intensity:int=30,
                 fill_color=(0,0,0,0)):

        if length < 0:
            raise Exception(f"Length must be a positive number. Got {length}")
        
        self.length = length
        self.direction = (-direction[0], direction[1])
        self.intensity = intensity
        self.fill_color = fill_color

    def augment(self, image):

        scratched_image = image.copy()

        def _scratch_line(start, length):
            get_x = lambda l: round(start[0] + self.direction[0] * l)
            get_y = lambda l: round(start[1] + self.direction[1] * l)

            points = [((get_x(i), get_y(i)), (get_x(i+length), get_y(i+length))) \
                      for i in range(length)]
            points = [(p1,p2) for p1, p2 in points if \
                      (valid_point(image, p1) and valid_point(image, p2))]

            [scratched_image.putpixel((p1), self.fill_color) for p1,p2 in points]
            [scratched_image.putpixel((p2), image.getpixel(p1)) for p1,p2 in points]
            
        for i in range(self.intensity):
            start = (randint(0, image.size[0]), randint(0, image.size[1]))
            length = randint(1, self.length)
            _scratch_line(start, length)

        return scratched_image

class IntensityLine(Augmentation):

    def __init__(self, intensity: float, line_loc: Optional[Tuple[int, int]]=None, 
                 line_width: int = 10, direction=(1,0)):
        super().__init__()

        if line_width <= 0:
            raise Exception(f"Line width must be a positive number. Got: {line_width}")
        
        self.intensity = intensity
        self.line_loc = line_loc
        self.line_width = line_width
        self.dirction = direction

    def augment(self, image):
        width, height = image.size
        
        background = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        background = np.asarray(
            Line((255, 255, 255, 255), self.line_loc, self.line_width, self.direction)(background))

        image = np.asarray(image, np.int32)
        image[background == 255] = image[background == 255] + self.intensity*255
        image[image < 0] = 0
        image[image > 255] = 255

        return Image.fromarray(image.astype(np.uint8))

class Line(Augmentation):

    def __init__(self, line_color:Tuple[int, int, int, int],
                 line_loc:Optional[Tuple[int, int]]=None, line_width:int=5, direction=(1,0)):
        super().__init__()

        if line_width < 1:
            raise Exception(f"Line width must be a positive number. Got: {line_width}")
        
        self.line_loc = line_loc
        self.line_width = line_width
        self.direction = direction
        self.line_color = line_color

    def augment(self, image):
        width, height = image.size

        if self.line_loc == None:
            self.line_loc = (width//2, height//2)

        diag_length = round(distance((0,0), (width, height)))

        start = (round(self.line_loc[0] - self.direction[0] * -(diag_length//2)), 
                 round(self.line_loc[1] + self.direction[1] * -(diag_length//2)))

        end = (round(start[0] - self.direction[0]*diag_length), (start[1] + self.direction[1]*diag_length))

        draw = ImageDraw.Draw(image)
        draw.line((start, end), self.line_color, self.line_width)

        return image

# TODO: Works for directions: (0,0), (1,0), (0,1) and (1,1). Not for direction in between.
class RepeatLine(Augmentation):
    def __init__(self, line_draw, between_line_width:int=5, skip_p=0):
        super().__init__()

        self.between_line_width = between_line_width
        self.line_draw = line_draw
        self.skip_p=skip_p

    def augment(self, image):
        width, height = image.size
        mid = [width//2, height//2]

        direction = self.line_draw.direction

        magnitude = sqrt(direction[0]**2 + direction[1]**2)
        direction = [direction[0] / magnitude, direction[1] / magnitude]


        start_point = [0,0]
        if direction[0] == 0:
            direction[0] = 1
            direction[1] = 0
            start_point[1] = mid[1]
        elif direction[1] == 0:
            direction[0] = 0
            direction[1] = 1
            start_point[0] = mid[0]
        guide_points = [start_point]

        for i in range(0, max(width, height), self.between_line_width):
            point = guide_points[-1]
            point = (
                point[0] + direction[0] * self.between_line_width,
                point[1] + direction[1] * self.between_line_width
            )
            guide_points.append(point)
        
        num_points = int((1-self.skip_p) * len(guide_points))
        guide_points = sample(guide_points, num_points)

        for line_loc in guide_points:
            self.line_draw.line_loc = line_loc
            image = self.line_draw(image)

        return image

class Crop(Augmentation):

    def __init__(self, size, center_offset: Tuple[int, int] = (0,0)):
        self.size = size
        self.offset = center_offset

    def augment(self, image):
        width, height = image.size
        mid = width//2, height//2

        left = mid[0] - (self.size[0] // 2) + self.offset[0]
        upper = mid[1] - (self.size[1] // 2) + self.offset[1]
        right = left + self.size[0]
        lower = upper + self.size[1]

        cropped = image.crop((left, upper, right, lower))

        return cropped
