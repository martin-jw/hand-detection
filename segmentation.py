import numpy as np

import skimage
from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage import future
from skimage import restoration
from skimage import morphology
from skimage import exposure
from skimage import measure
from skimage import color
from skimage import draw

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

from util.debugutil import create_debug_fig


def create_bin_img_slic(image, segments=500, ):

    image = restoration.denoise_tv_chambolle(image, weight=0.05, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=500)
    g = future.graph.rag_mean_color(image, labels1)

    labels2 = future.graph.cut_threshold(labels1, g, 0.06)

    if __debug__:
        create_debug_fig(color.label2rgb(labels2, image), "Threshold Cut")

    bin_test = np.zeros((300, 400))
    for r in measure.regionprops(labels2):
        rr, cc = zip(*r.coords)
        bin_test[rr, cc] = 1

    return bin_test


def create_bin_img_otsu(image):

    image = color.rgb2gray(image)

    val = filters.threshold_otsu(image)
    img = image.copy() >= val

    img = skimage.img_as_float(img)

    return img
