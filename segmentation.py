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
    """A background segmentation algorithm. Segments the image using Simple Linear Iterative Clustering and a graph cut. One
    region is chosen as the background and the rest is chosen as foreground.

    Keyword arguments:
    image -- The image to segment.
    segments -- The number of segments SLIC should segment into. Default is 500.
    thresh -- The threshold for the graph cut. Default is 0.1."""

    image = restoration.denoise_tv_chambolle(image, weight=0.04, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=segments)
    g = future.graph.rag_mean_color(image, labels1)

    fig, axes = plt.subplots(ncols=2, figsize=(8, 8))
    ax = axes.ravel()

    labels2 = future.graph.cut_threshold(labels1, g, thresh)

    ax[0].imshow(color.label2rgb(labels1, image))
    ax[0].set_title("SLIC Clustering")

    ax[1].imshow(color.label2rgb(labels2, image))
    ax[1].set_title("Graph Cut")

    for a in ax:
        a.axis('off')

    fig.tight_layout()

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
    labels = measure.label(img, background=2)

    mx_region = max(measure.regionprops(labels), key=lambda r: r.area)

    rr, cc = list(zip(*mx_region.coords))
    if np.mean(img[rr, cc]) == 1:
        img = np.invert(img)

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

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(denoised, cmap='gray', interpolation='nearest')
    ax[0].set_title('Denoised')

    ax[1].imshow(gradient, cmap='spectral', interpolation='nearest')
    ax[1].set_title('Local Gradient')

    ax[2].imshow(markers, cmap='spectral', interpolation='nearest')
    ax[2].set_title('markers')

    ax[3].imshow(img, cmap='gray', interpolation='nearest')
    ax[3].imshow(labels, cmap='spectral', interpolation='nearest', alpha=.7)
    ax[3].set_title("Segmented")

    for a in ax:
        a.axis('off')

    fig.tight_layout()

    return binary, False
