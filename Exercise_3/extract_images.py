# Take the images in .jpg, coordinates in .svg and extract binarized chunks of images as masked numpy arrays

# Imports
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave as ims
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

# data paths

LOCATIONS = 'data/input_documents/locations/'
IMAGES = 'data/input_documents/images/'
TRANSCRIPT = 'data/input_documents/transcription.txt'
TEMP = 'data/temp_images/'
OUTPUT = 'data/cropped_words/'

# choose which methods to use for backgroun shizzl removal
# choose 'sauvola', otsu', 'niblack'

method = 'sauvola'

# create all folders
for directory in [LOCATIONS, IMAGES, TEMP, OUTPUT]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# =============================================#
#             Functions definitions            #
# =============================================#



def ExtractPaths(location_file):
    """
    Extract all path from svg file, along with ID
    :param location_file: svg file
    :return: 
    """
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
    image = plt.imread(image_file)
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


def RemoveBackground(img, method, window_size = 25, k = 0.8):
    """
    Create a binary image separating foreground from background
    :param img: a numpy array
    :param method: one of ['otsu', 'niblack', 'sauvola']
    :param window_size: size of neighborhood used to define threshold, used in ['niblack', 'sauvola']
    :param k: used to tune local threshold, used in ['niblack', 'sauvola']
    :return: numpy array, representing a binary image
    """
    image = np.copy(img)
    if method == 'otsu':
        threshold = filters.threshold_otsu(img)
    elif method == 'niblack':
        threshold = filters.threshold_niblack(img, window_size, k)
    elif method == 'sauvola':
        threshold = filters.threshold_sauvola(img)
    image[image <= threshold] = 0
    image[image >= threshold] = 255
    return image


# =============================================#
#                Apply functions               #
# =============================================#

# Extract all locations for all files

locations = {}
for file in os.listdir(LOCATIONS):
    temp = ExtractPaths(LOCATIONS + file)
    locations.update(temp)
del temp, file

# =============================================
# Get all words ids
IDs = []
with open(TRANSCRIPT) as f:
    for line in f:
        IDs.append(line.split(sep=" ")[0])
IDs = [ids.split(sep="-") for ids in IDs]

# =============================================
# Create and save a filtered version of the image files
# /!\  Create folder 'filtered_images' in folder 'data' and  /!\
# /!\     place two subfolders 'sauvola' and 'otsu' in it    /!\

img_names = [str(i) for i in range(270, 280)]
img_names.extend([str(i) for i in range(300, 305)])


img_name = [i + '.jpg' for i in img_names]
for im in img_name:
    image = Image.open(IMAGES + im)
    image = np.asarray(image)
    image = RemoveBackground(img=image, method=method)
    ims(name=TEMP + '/' + im.replace('jpg', 'png'), arr=image)

# =============================================
# Produce a cropped image for each word, remove background and save


for ids in IDs:
    path = tuple(ids)
    temp = ExtractImage(TEMP + '/' + path[0] + '.png', locations, path,
                            crop=True)
    # Image is rectangular, crop mask is non-rectangular polygon so need to set undesired px as white
    temp = np.ma.filled(temp, 1)
    ims(name=OUTPUT + '/' + path[0] + '-' + path[1] + '-' + path[2]
                         + '.png', arr=temp)

'''
# =============================================#
#       Example section and visualization      #
# =============================================#

# Polygon path
# a = locations[('270','01','01')]
# temp = PathToPolygon(a)
# temp

# =============================================
# Extract a word form an image
file = '271'
line = '22'
word = '04'
path = (file, line, word)
temp = ExtractImage(IMAGES + file + '.jpg', locations, path, crop=True)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
plt.show()
del path

# =============================================
# Compare background removal methods,
# There's a scikit-image builtin function for simpler thresholds but results are not as good,
# see based on http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html

# Source code based on http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_niblack_sauvola.html

for im in [IMAGES + i + '.jpg' for i in img_names][0:3]:
    image = Image.open(im)
    image = np.asarray(image)
    binary_global = image > threshold_otsu(image)

    window_size = 25
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    plt.figure(figsize=(8, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Otsu Threshold')
    plt.imshow(binary_global, cmap=plt.get_cmap('gray'))
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(binary_niblack, cmap=plt.get_cmap('gray'))
    plt.title('Niblack Threshold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary_sauvola, cmap=plt.get_cmap('gray'))
    plt.title('Sauvola Threshold')
    plt.axis('off')

    plt.show()'''
