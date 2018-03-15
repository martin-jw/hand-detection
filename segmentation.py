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
from skimage.filters import rank

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

from util.debugutil import create_debug_fig


def create_bin_img_slic(image, segments=500, thresh=0.1):

    image = restoration.denoise_tv_chambolle(image, weight=0.04, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=segments)
    g = future.graph.rag_mean_color(image, labels1)

    labels2 = future.graph.cut_threshold(labels1, g, thresh)

    if __debug__:
        create_debug_fig(color.label2rgb(labels2, image), "Threshold Cut")

    bin_test = np.zeros((300, 400))
    for r in measure.regionprops(labels2):
        rr, cc = zip(*r.coords)
        bin_test[rr, cc] = 1

    return bin_test, False


def create_bin_img_otsu(image):

    image = color.rgb2gray(image)

    val = filters.threshold_otsu(image)
    img = image.copy() >= val

    img = skimage.img_as_float(img)

    return img, True

def create_bin_img_watershed(image):

    img = color.rgb2gray(image)
    denoised = rank.median(img, morphology.disk(1))

    markers = rank.gradient(denoised, morphology.disk(4)) < 20
    markers = ndi.label(markers)[0]

    gradient = rank.gradient(denoised, morphology.disk(2))

    labels = morphology.watershed(gradient, markers)
    binary = np.zeros((labels.shape[0], labels.shape[1]), dtype='float64')

    for r in measure.regionprops(labels):
        if r.label != 1:
            rr, cc = zip(*r.coords)
            binary[rr, cc] = 1

    return binary, False