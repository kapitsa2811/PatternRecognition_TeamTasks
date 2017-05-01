import matplotlib.pyplot as plt
import numpy as np
from PIL import Image




def calculateFeatureVector(filename):
    img = np.asarray(Image.open(filename).resize((200, 200)))

    G = np.zeros((200, 200, 3))

    # Where we set the RGB for each pixel
    G[img > 150] = True
    G[not True] = False
    img = G

    plt.imshow(img)
    plt.show()
    return img


def window_stack(a, stepsize=1, width=1):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

imt = calculateFeatureVector("test"+'.jpg')
window_stack(imt,1,1)

