import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image




def calculateFeatureVector(filename):
    img = np.asarray(Image.open(filename).resize((200, 200)))
    plt.imshow(img)
    plt.show()

calculateFeatureVector("test"+'.jpg')

