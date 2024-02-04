import argparse

from pathlib import Path

from CraftPatternGenerator.CraftPatternGenerator import CraftPatternGenerator
from CraftPatternGenerator.input import preset_colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image to convert')
    parser.add_argument('origin_path', type=str, action='store',
                        help='image to convert')
    parser.add_argument('color_chart', type=str, action='store',
                        help='name of the chart to use, should match with one of those in input.py')

    colors = parser.add_mutually_exclusive_group()
    colors.add_argument('--crop-color-rgb', dest='crop_color_rgb', action='store', type=list, nargs="+",
                        help='The color to use for crop the image if not given no crop will be performed')
    colors.add_argument('--crop-color-name', dest='crop_color_name', action='store', type=str,
                        help='The color to use for crop the image if not given no crop will be performed')
    parser.add_argument('--crop-tolerance', type=int, dest='crop_tolerance', action='store', default=0,
                        help='The tolerance in color space to match for crop, 0 by default,'
                             ' color space is in lab by default')
    parser.add_argument('--target-size', dest='target_size', action='store', type=int, nargs="+",
                        default=[-1],
                        help='The size of the output image,'
                             ' if Not given no resize will be performed,'
                             ' if only one int given, the biggest side will be resized to the target and the tinier'
                             ' will be resize proportionaly, if a tuple is given the two side will be resise'
                             ' to match the tuple')
    args = parser.parse_args()
    origin_path = args.origin_path
    color_chart = args.color_chart
    crop_color = None
    if getattr(args, "crop_color_rgb", None):
        if len(args.crop_color_rgb) != 3:
            raise Exception("crop_color_rgb should be a list of 3 integers")
        try:
            crop_color = tuple(int(color) for color in args.crop_color)
        except ValueError:
            raise Exception("crop_color_rgb list should only contains integer")
        if any( 0 > color > 255 for color in crop_color):
            raise Exception("crop_color_rgb list should only contains integer between 0 and 255")
    if getattr(args, "crop_color_name", None):
        if args.crop_color_name not in preset_colors:
            raise Exception(f'crop_color_name should be one of {", ".join(color for color in preset_colors)}')
        crop_color = preset_colors[args.crop_color_name]
    crop_tolerance = args.crop_tolerance
    if len(args.target_size) != 1 and len(args.target_size) != 2:
        raise Exception(f"target_size should contain exactly 1 or 2 values, currently: {args.target_size}")
    target_size = None
    if len(args.target_size) == 2:
        target_size = tuple(args.target_size)
    if len(args.target_size) == 1:
        target_size = args.target_size[0]

    converter = CraftPatternGenerator(
        Path(args.origin_path),
        color_chart,
        crop_color,
        crop_tolerance,
        target_size
    )
    converter.run()
