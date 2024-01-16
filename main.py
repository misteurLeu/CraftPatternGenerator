import argparse

from pathlib import Path
from typing import Union

from CraftPatternGenerator.CraftPatternGenerator import CraftPatternGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image to convert')
    parser.add_argument('origin_path', type=str, action='store',
                        help='image to convert')
    parser.add_argument('color_chart', type=str, action='store',
                        help='name of the chart to use, should match with one of those in input.py')

    parser.add_argument('--crop-color', dest='crop_color', action='store', type=int, nargs="+",
                        default=[None], help='The color to use for crop the image if not given no crop will be performed')
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
    if len(args.crop_color) != 1 and len(args.crop_color) != 3:
        raise Exception(f"Crop_color should contain exactly 1 or 3 values, currently: {args.crop_color}")
    if len(args.crop_color) == 3:
        crop_color = tuple(args.crop_color)
    if len(args.crop_color) == 1:
        crop_color = args.crop_color[0]
    crop_tolerance = args.crop_tolerance
    if len(args.target_size) != 1 and len(args.target_size) != 2:
        raise Exception(f"target_size should contain exactly 1 or 2 values, currently: {args.target_size}")
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
