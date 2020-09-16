import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.ndimage

def plot_graph(metrics):
    plt.plot(metrics)
    plt.ylabel('Performance Metrics')
    plt.show()


def display_images(fixed_image, moving_image):
    """

    :param fixed_image: First image
    :param moving_image: Second image
    :return: displays two images next to each other
    """
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(fixed_image), cmap=plt.cm.Greys_r)
    plt.title('fixed image')
    plt.axis('off')

    ## Draw the moving image
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(moving_image), cmap=plt.cm.Greys_r)
    plt.title('moving image')
    plt.axis('off')

    plt.show()

def itk_show(img, title=None):
    nda = sitk.GetArrayFromImage(img)
    plt.imshow(nda, cmap=plt.cm.Greys_r)
    plt.axis('off')

    if title:
        plt.title(title)

    plt.show()


def arr_show(arr, title=None):
    plt.imshow(arr, cmap=plt.cm.Greys_r)
    plt.axis('off')

    if title:
        plt.title(title)

    plt.show()

### 3D imaging testing

