import colorsys
import math
import pandas as pd

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing import Optional, Union

from .input import (
    COLORS_CHARTS,
    COLORS_CHART_HEADER,
    preset_colors,
    COLORS_CODES
)


class CraftPatternGeneratorException(Exception):
    pass


class BadBoundingBoxesException(CraftPatternGeneratorException):
    pass


class IndexException(CraftPatternGeneratorException):
    pass


class ColorsCodesException(CraftPatternGeneratorException):
    pass


RGB_SCALE = 255
CMYK_SCALE = 100


def rgb_to_cmyk(input_color: tuple) -> tuple:
    """Convert colors from rgb space to cmyk."""
    r, g, b = (input_color[0], input_color[1], input_color[2])
    if (r, g, b) == (0, 0, 0):
        # black
        return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE


def rgb2lab(input_color: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert colors from rgb space to lab."""
    rgb = list()

    for color in input_color:
        color = float(color) / 255
        color = ((color + 0.055) / 1.055)**2.4 if color > 0.04045 else color / 12.92
        rgb.append(color * 100)

    x = rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805
    y = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
    z = rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505

    xyz = [
        float(round(x, 4)) / 95.047, # ref_X =  95.047   Observer= 2°, Illuminant= D65
        float(round(y, 4)) / 100.0,  # ref_Y = 100.000
        float(round(z, 4)) / 108.883 # ref_Z = 108.883
    ]

    for index, value in enumerate(xyz):
        if value > 0.008856:
            value = value ** (1/3)
        else:
            value = (7.787 * value) + (16 / 116)
        xyz[index] = value

    l = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])

    lab = (round(l, 4), round(a, 4) + 128, round(b, 4) + 128)

    return lab


class CraftPatternGenerator:
    """Class to generate a pattern based on a source image for craft hobbies."""
    def __init__(self,
                 image_path: Path,
                 chart_type: str,
                 crop: Optional[tuple] = None,
                 crop_tolerance: Optional[float] = None,
                 target_size: Optional[int] = None,
                 saturation: Optional[float] = None):
        """Initialize the CraftPatternGenerator class."""
        self.chart = self.download_chart(chart_type)
        self.saturation = saturation
        self.image_in = self.load_image(image_path)
        self.image_in = self.auto_crop(self.image_in, crop, crop_tolerance or 0) if crop is not None else self.image_in
        self.image_in = self.image_in
        # merge the near colors
        if target_size != -1:
            target_size = self.get_new_size_value(target_size)
            self.image_in.palette = None
            self.image_in = self.image_in.resize(target_size, Image.BICUBIC)
        self.image_path = image_path
        
    def get_color_from_index(self, index: int) -> tuple[int, int, int, int]:
        return (
            self.chart.iloc[index]['r'],
            self.chart.iloc[index]['g'],
            self.chart.iloc[index]['b'],
            255
        )
        
    def get_new_size_value(self, target_size: Union[tuple[int, int], int]) -> tuple[int, int]:
        """Calculate the new size for the image to process."""
        if isinstance(target_size, tuple):
            return target_size
        x, y = self.image_in.size
        if x > y:
            return target_size, int(y * target_size / x)
        return int(x * target_size / y), target_size

    def replace_color(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """Replace the color from image with the nearest in the chart."""
        w, h = image.size
        colors_counted = defaultdict(lambda: 0)
        for i in range(w):
            print(f"processing column {i} on {w}")
            for j in range(h):
                curent_color = image.getpixel((i, j))
                if curent_color[3] == 0:
                    continue
                neerest_color = self.get_neerest_color(curent_color)
                colors_counted[neerest_color] += 1
                color = self.get_color_from_index(neerest_color)
                image.putpixel((i, j), color)

        return image, colors_counted
    
    def replace_color_a_to_b(self, image: Image.Image, color_a: tuple, color_b: tuple) -> tuple[Image.Image, dict]:
        """Replace the color from image with the nearest in the chart."""
        w, h = image.size
        colors_counted = defaultdict(lambda: 0)
        for i in range(w):
            for j in range(h):
                curent_color = image.getpixel((i, j))
                if curent_color[3] == 0:
                    continue
                if curent_color == color_a:
                    image.putpixel((i, j), color_b)
                    curent_color = color_b
                neerest_color = self.get_neerest_color(curent_color)
                colors_counted[neerest_color] += 1
        return image, colors_counted

    def reduce_color(self, image: Image.Image, color_limit: int, colors_index: dict) -> tuple[Image.Image, dict]:
        """Reduce the color numbers in the image by merging the nearest colors."""
        if color_limit < 2:
            return image, colors_index

        while any(val < color_limit for val in colors_index.values()):
            colors_matching = {}
            for color_key1 in colors_index.keys():
                color1 = self.get_color_from_index(color_key1)
                colors_matching[color_key1] = {
                    "color": color1,
                    "neerest_color": color1,
                    "neerest_color_key": color_key1,
                    "distance": math.inf
                }
                for color_key2 in colors_index.keys():
                    if color_key1 == color_key2:
                        continue
                    color2 = self.get_color_from_index(color_key2)
                    distance = self.get_color_distance(color1, color2)
                    if distance < colors_matching[color_key1]["distance"]:
                        colors_matching[color_key1]["neerest_color"] = color2
                        colors_matching[color_key1]["distance"] = distance
                        colors_matching[color_key1]["neerest_color_key"] = color_key2
            color_keys = colors_matching.keys()
            color_keys = sorted(color_keys, key=lambda x: colors_index[x])
            print(f"color_keys: {color_keys}")
            color_key = color_keys.pop(0)
            color1 = colors_matching[color_key]["color"]
            color2 = colors_matching[color_key]["neerest_color"]
            color1_count = colors_index[color_key]
            color2_count = colors_index[colors_matching[color_key]["neerest_color_key"]]
            color_a, color_b = (color1, color2) if color1_count < color2_count else (color2, color1)
            print(f"merge {color_key} ({color1_count} counted) with {colors_matching[color_key]['neerest_color_key']} ({color2_count} counted)")
            image, colors_index = self.replace_color_a_to_b(image, color_a, color_b)

        return image, colors_index

    def run(self):
        """Launch the image processing"""
        image_colored, colors_avaible = self.replace_color(self.image_in)
        image_colored, colors_avaible = self.reduce_color(image_colored, 30, colors_avaible)
        self.output_image(image_colored, f'{self.image_path.parent}/{self.image_path.stem}', list(colors_avaible.keys()))

    def output_image(self, image_to_out: Image.Image, output_path: str, colors: list):
        """Export the final image as a usable pattern for crafting activities."""
        if len(colors) > len(COLORS_CODES):
            raise ColorsCodesException(f"Not enought color code avaible in COLORS_CODES"
                                       f" {len(COLORS_CODES)} avaible, but {len(colors)} needed")
        w, h = image_to_out.size
        squares_sizes = 10
        font_squares_size = squares_sizes - 2
        font_resume_size = int(h * 10 / len(colors)) - 2
        square_font = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", font_squares_size)
        resume_font = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", font_resume_size)
        image_out = Image.new(
            'RGB',
            (w * squares_sizes + font_resume_size * 10, h * squares_sizes),
            preset_colors["white"]
        )
        preview_size = max(w, h)
        preview_displacement_h = 0 if w <= h else (w - h) // 2
        preview_displacement_w = 0 if h <= w else (h - w) // 2
        image_preview = Image.new(
            'RGBA',
            (preview_size, preview_size),
            preset_colors["transparent"]
        )

        draw = ImageDraw.Draw(image_out)
        colors_count = defaultdict(lambda: 0)
        for i in range(w):
            for j in range(h):
                curent_color = image_to_out.getpixel((i, j))
                font_color = self.get_contrasted_color(curent_color)
                if curent_color[3] == 0:
                    continue
                image_preview.putpixel((i + preview_displacement_w, j + preview_displacement_h), curent_color)
                draw.rectangle(
                    (i * squares_sizes, j * squares_sizes,
                     (i + 1) * squares_sizes, (j + 1) * squares_sizes),
                    fill=curent_color,
                    outline=font_color,
                    width=1
                )
                index = self.get_color_index(colors, curent_color)
                colors_count[index] += 1
                draw.text(
                    (i * squares_sizes + 3, j * squares_sizes),
                    f"{COLORS_CODES[index]}",
                    font_color,
                    font=square_font
                )
        for index, color_index in enumerate(colors):
            color = (
                self.chart.iloc[color_index]['r'],
                self.chart.iloc[color_index]['g'],
                self.chart.iloc[color_index]['b']
            )
            font_color = self.get_contrasted_color(color)
            draw.ellipse((
                w * squares_sizes + 1 * font_resume_size,
                index * font_resume_size,
                w * squares_sizes + 2 * font_resume_size,
                (index + 1) * font_resume_size
            ),
                fill=color)
            draw.text(
                (w * squares_sizes + 1 * font_resume_size, index * font_resume_size),
                f"{COLORS_CODES[index]}",
                font_color,
                font=resume_font
            )
            draw.text(
                (w * squares_sizes + 2 * font_resume_size, index * font_resume_size),
                f" - {self.chart.iloc[color_index]['ref']}",
                (15, 15, 15),
                font=resume_font
            )
            draw.text(
                (w * squares_sizes + (2 + len(self.chart.iloc[color_index]['ref'])) * font_resume_size,
                 index * font_resume_size),
                f" - x{colors_count[index]}",
                (15, 15, 15),
                font=resume_font
            )
        image_out.save(f'{output_path}-{preview_size}-out.png')
        image_preview.save(f'{output_path}-{preview_size}-preview.png')

    def get_color_index(self, colors: list, color: tuple[int, int, int]) -> int:
        """Given a color, find his index inside the selected chart."""
        r, g, b = (color[0], color[1], color[2])
        for index, color_index in enumerate(colors):
            if self.chart.iloc[color_index]['r'] == r and\
                    self.chart.iloc[color_index]['g'] == g and\
                    self.chart.iloc[color_index]['b'] == b:
                return index
        else:
            raise IndexException(f"Color not found in list")

    def auto_crop(self, to_crop: Image.Image, color: tuple = preset_colors['transparent'], tolerance: float = 0):
        """Remove the first & last lines & column that contain only the given color, with some distance tolerances."""
        width, height = to_crop.size
        crop_box = [0, 0, width, height]
        # left
        for i in range(width):
            if not all([
                self.get_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for j in range(height)
            ]):
                crop_box[0] = i
                break
        # top
        for j in range(height):
            if not all([
                self.get_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for i in range(width)
            ]):
                crop_box[1] = j
                break
        # right
        for i in range(width - 1, -1, -1):
            if not all([
                self.get_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for j in range(height)
            ]):
                crop_box[2] = i
                break
        # bottom
        for j in range(height - 1, -1, -1):
            if not all([
                self.get_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for i in range(width)
            ]):
                crop_box[3] = j
                break
        # if crop all image raise an error
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise BadBoundingBoxesException(f"Bounding boxe {crop_box} have illegal values")
        to_crop.crop()
        return to_crop.crop(tuple(crop_box))

    def load_image(self, image_path: Path, mode: str = 'RGBA') -> Image:
        """Load an image in memory."""
        img = Image.open(image_path)
        rgb_image = img.convert(mode=mode)
        return rgb_image

    def download_chart(self, chart_type: str) -> pd.DataFrame:
        """Download the main chart and filter it to keep only keys from the selected chart."""
        if COLORS_CHARTS[chart_type]['parent'] is None:
            return pd.read_csv(
                COLORS_CHARTS[chart_type]['path'],
                sep=',',
                encoding='utf-8',
                names=COLORS_CHART_HEADER
            )
        chart = self.download_chart(COLORS_CHARTS[chart_type]['parent'])
        chart = chart[chart['ref'].isin(COLORS_CHARTS[chart_type]['avaible_keys'])]
        chart.reset_index(drop=True, inplace=True)
        return chart

    @lru_cache
    def get_color_distance(self, color1: tuple, color2: tuple) -> float:
        """Process the color chart in the color space give in parameter."""
        r = (color1[0] + color2[0]) / 2
        delta_R = (color1[0] - color2[0]) ** 2
        delta_G = (color1[1] - color2[1]) ** 2
        delta_B = (color1[2] - color2[2]) ** 2
        distance = (2 + r / 256) * delta_R + 4 * delta_G + (2 + ((255 - r) / 256)) * delta_B
        return math.sqrt(distance)

    @lru_cache
    def get_neerest_color(self, color: tuple):
        """Search for the nearest color in color chart."""
        return self.chart.apply(lambda x: self.get_color_distance(
            (x['r'], x['g'], x['b']),
            color
        ), axis=1).idxmin()

    @lru_cache
    def get_contrasted_color(self, color: tuple[int, int, int]):
        """Return the best color to use as text for a given background color, to have the best contrast."""
        lum = ((0.299 * color[0]) + (0.587 * color[1]) + (0.114 * color[2])) / 255

        return preset_colors['white'] if lum < 0.5 else preset_colors['black']
