# Take the images jpg, coordinates svg and extract chunks of images
import numpy as np
import matplotlib as plt

from xml.dom import minidom
from svg.path import parse_path

svg_dom = minidom.parseString('./ground-truth/locations/270.svg')

path_strings = [path.getAttribute('d') for path in svg_dom.getElementsByTagName('path')]

for path_string in path_strings:
    path_data = parse_path(path_string)


def ExtractCoordinates(location_file):
    """
    Extract all path from svg file, along with ID
    :param location_file: svg file
    :return: 
    """

    import re
    pattern = r'''^
    \s+
    <path\
    fill="none"\
    d=(?P<path>".*?")\
    '''
    out = []
    pattern = re.compile(pattern, re.VERBOSE)
    with open(location_file) as f:
        for line in f:
            result = pattern.search(line)
            if result:
                print("match")
                path = result.group('path')
                out.append(path)
    return out

temp = ExtractCoordinates('./Exercise_3/data/ground-truth/locations/270.svg')