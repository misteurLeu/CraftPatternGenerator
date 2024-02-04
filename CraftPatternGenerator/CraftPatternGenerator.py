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
        self.image_in = self.load_image(image_path)
        self.image_in = self.auto_crop(self.image_in, crop, crop_tolerance or 0) if crop is not None else self.image_in
        self.saturation = saturation
        if target_size != -1:
            target_size = self.get_new_size_value(target_size)
            self.image_in = self.image_in.resize(target_size, Image.NEAREST)
        if self.saturation is not None:
            self.saturate()
        self.image_path = image_path

    def get_new_size_value(self, target_size: Union[tuple[int, int], int]) -> tuple[int, int]:
        """Calculate the new size for the image to process."""
        if isinstance(target_size, tuple):
            return target_size
        x, y = self.image_in.size
        if x > y:
            return target_size, int(y * target_size / x)
        return int(x * target_size / y), target_size

    def saturate(self):
        """Increase the image color's saturation."""
        enhancer = ImageEnhance.Color(self.image_in)
        self.image_in = enhancer.enhance(self.saturation)

    def replace_color(self, image: Image.Image) -> tuple[Image.Image, set]:
        """Replace the color from image with the nearest in the chart."""
        w, h = image.size
        colors_index = set()
        for i in range(w):
            print(f"processing column {i} on {w}")
            for j in range(h):
                curent_color = image.getpixel((i, j))
                if curent_color[3] == 0:
                    continue
                neerest_color = self.get_neerest_color(curent_color)
                colors_index.add(neerest_color)
                color = (
                    self.chart.iloc[neerest_color]['r'],
                    self.chart.iloc[neerest_color]['g'],
                    self.chart.iloc[neerest_color]['b'],
                    255
                )
                image.putpixel((i, j), color)
        return image, colors_index

    def run(self):
        """Launch the image processing"""
        image_colored, colors_avaible = self.replace_color(self.image_in)
        self.output_image(image_colored, f'{self.image_path.parent}/{self.image_path.stem}-out.jpg', list(colors_avaible))

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
            (240, 240, 240, 0)
        )
        draw = ImageDraw.Draw(image_out)
        colors_count = defaultdict(lambda: 0)
        for i in range(w):
            for j in range(h):
                curent_color = image_to_out.getpixel((i, j))
                font_color = self.get_contrasted_color(curent_color)
                if curent_color[3] == 0:
                    continue
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
        image_out.save(output_path)

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
                self.euclidian_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for j in range(height)
            ]):
                crop_box[0] = i
                break
        # top
        for j in range(height):
            if not all([
                self.euclidian_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for i in range(width)
            ]):
                crop_box[1] = j
                break
        # right
        for i in range(width - 1, -1, -1):
            if not all([
                self.euclidian_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
                for j in range(height)
            ]):
                crop_box[2] = i
                break
        # bottom
        for j in range(height - 1, -1, -1):
            if not all([
                self.euclidian_color_distance(to_crop.getpixel((i, j)), color) <= tolerance
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
    def euclidian_color_distance(self, color1: tuple, color2: tuple, convert_space='') -> float():
        """Process the color chart in the color space give in parameter."""
        if convert_space == 'cmyk':
            color1 = rgb_to_cmyk(color1)
            color2 = rgb_to_cmyk(color2)
        elif convert_space == 'lab':
            color1 = rgb2lab(color1)
            color2 = rgb2lab(color2)
        elif convert_space == 'hsv':
            color1 = colorsys.rgb_to_hsv(color1[0] / 255, color1[1] / 255, color1[2] / 255)
            color2 = colorsys.rgb_to_hsv(color2[0] / 255, color2[1] / 255, color2[2] / 255)
        elif convert_space == 'hls':
            color1 = colorsys.rgb_to_hls(color1[0] / 255, color1[1] / 255, color1[2] / 255)
            color2 = colorsys.rgb_to_hls(color2[0] / 255, color2[1] / 255, color2[2] / 255)
        elif convert_space == 'yiq':
            color1 = colorsys.rgb_to_yiq(color1[0] / 255, color1[1] / 255, color1[2] / 255)
            color2 = colorsys.rgb_to_yiq(color2[0] / 255, color2[1] / 255, color2[2] / 255)
        distance = sum([(elem2 - elem1)**2 for elem1, elem2 in zip(color1, color2) if elem2 is not None])
        distance = math.sqrt(distance)
        return distance

    @lru_cache
    def get_neerest_color(self, color: tuple):
        """Search for the nearest color in color chart."""
        return self.chart.apply(lambda x: self.euclidian_color_distance(
            (x['r'], x['g'], x['b']),
            color,
            'lab'
        ), axis=1).idxmin()

    @lru_cache
    def get_contrasted_color(self, color: tuple[int, int, int]):
        """Return the best color to use as text for a given background color, to have the best contrast."""
        lum = ((0.299 * color[0]) + (0.587 * color[1]) + (0.114 * color[2])) / 255

        return preset_colors['white'] if lum < 0.5 else preset_colors['black']
