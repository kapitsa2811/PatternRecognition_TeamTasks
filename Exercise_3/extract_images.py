# Take the images in .jpg, coordinates in .svg and extract chunks of images as masked numpy arrays
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


def ExtractImage(image_file, locations, path_id, crop = True):
    """
    Extract the part of an image delimited by path
    :param image_file: a jpg file
    :param locations: a dictionary with path ids as keys in format (XXX-YY-ZZ) and path as items
    :param path_id: id of the chunk to extract
    :param crop: If False returns image of same size as original, if True remove rows and columns that are fully masked
    :return: a masked numpy array with only the specified path unmasked
    """
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.open(image_file)
    image = np.asarray(image)
    polyg = PathToPolygon(locations[path_id])
    # Create an image filled with 1, set 0 in the region delimited by the polygon
    maskIm = Image.new('L', (image.shape[1], image.shape[0]), 1)
    ImageDraw.Draw(maskIm).polygon(polyg, outline=0, fill=0)
    # Use the previous function as a mask
    mask = np.array(maskIm)
    masked_image = np.ma.array(image, mask=mask)
    if crop:
        masked_image = masked_image[~np.all(masked_image == 0, axis=1), :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=0)]
    return masked_image


# =============================================
# Extract all locations for all files
import os
import numpy as np
locations = {}
for file in os.listdir('./Exercise_3/data/ground-truth/locations/'):
    temp = ExtractPaths('./Exercise_3/data/ground-truth/locations/' + file)
    locations.update(temp)
del temp, file

# Access
file = '271'
line = '22'
word = '04'

path = (file, line, word)
from matplotlib import pyplot as plt
temp = ExtractImage('./Exercise_3/data/images/' + file + '.jpg', locations, path, crop=True)
plt.imshow(temp, cmap='gray')
plt.show()
del path


# Produce a cropped image for each word, remove background and save
# Get all word ids
IDs = []
with open('./Exercise_3/data/ground-truth/transcription.txt') as f:
    for line in f:
        IDs.append(line.split(sep = " ")[0])
IDs = [ids.split(sep = "-") for ids in IDs]

# ExtractImage for each word and save
import scipy.misc
threshold = 15 # Threshold to remove background percentile of the polygon; 15% looks like a general good compromise
for ids in IDs:
    path = tuple(ids)
    temp = ExtractImage('./Exercise_3/data/images/' + path[0] + '.jpg', locations, path, crop=True)
    threshold = np.percentile(temp, 15)
    # Image is rectangular, crop mask is polygone so need to set undesired px as white
    temp[temp.mask == 1] = 255
    # Remove background
    temp[temp >= threshold] = 255
    # Amplify signal (turned off for the moment as it also amplifies remaining background)
    # temp[np.logical_and(temp.mask == 0, temp < threshold )] = 0
    scipy.misc.toimage(temp, cmin=0.0, cmax=...).save('Exercise_3/data/cropped_words/' + path[0] + '-' + path[1] + '-' + path[2] + '.jpg')
