# Take the images jpg, coordinates svg and extract chunks of images
# =============================================


def ExtractPaths(location_file):
    """
    Extract all path from svg file, along with ID
    :param location_file: svg file
    :return: 
    """

    import re
    pattern = r'''^
    \s+
    <path\s
    fill=".*?"\s
    d="(?P<path>.*?)"\s
    stroke-width=".*?"\s
    id="(?P<id>.*?)"\s
    stroke=".*?"
    />
    $
    '''
    out = {}
    pattern = re.compile(pattern, re.VERBOSE)
    with open(location_file) as f:
        for line in f:
            result = pattern.search(line)
            if result:
                path = result.group('path')
                id = result.group('id')
                id = tuple(id.split('-'))
                out[id] = path
    return out


# =============================================


def PathToPolygon(path):
    """
    Convert an SVG segments path to a polygon list, use as mask to isolate chunk of image
    :param path: svg path (characters string) that looks like 'M 1131.00 600.00 L 1151.00 599.00 L 1211.00 599.00 Z'
    :return: A list of tuples with the coordinates of the vertices of the corresponding polygon
    """
    # Trim start and end of path
    path = path.lstrip('M ').rstrip(' Z')
    # Isolate segments start and end
    path = path.split('L ')
    path = [pair.rstrip() for pair in path]
    # Convert to float
    polygon = [pair.split() for pair in path]
    polygon = [(float(pair[0]), float(pair[1])) for pair in polygon]
    return polygon


# Example
# a = locations[('270','01','01')]
# temp = PathToPolygon(a)
# temp

# =============================================


def ExtractImage(image_file, locations, path_id):
    """
    Extract the part of an image delimited by path
    :param image_file: a jpg file
    :param locations: a dictionary with path ids as keys in format (XXX-YY-ZZ) and path as items
    :param path_id: id of the chunk to extract
    :return: a masked numpy array with only the specified path unmasked
    """
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.open(image_file)
    image = np.asarray(image)
    polyg = PathToPolygon(locations[path_id])
    # Create an image filled with 1, set 0 in the region delimited by the polygon
    maskIm = Image.new('L', (image.shape[1], image.shape[0]), 1)
    ImageDraw.Draw(maskIm).polygon(polyg, outline=1, fill=0)
    # Use the previous function as a mask
    mask = np.array(maskIm)
    masked_image = np.ma.array(image, mask=mask)
    return masked_image


# =============================================
# Extract all locations for all files
import os
locations = {}
for file in os.listdir('./Exercise_3/data/ground-truth/locations/'):
    temp = ExtractPaths('./Exercise_3/data/ground-truth/locations/' + file)
    locations.update(temp)
del temp, file

# Access
file = '271'
line = '09'
word = '08'

path = (file, line, word)
from matplotlib import pyplot as plt
temp = ExtractImage('./Exercise_3/data/images/' + file + '.jpg', locations, path)
plt.imshow(temp, cmap='gray')
plt.show()
del path