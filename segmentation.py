import numpy as np

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


def create_bin_img_slic(image):

    image = restoration.denoise_tv_chambolle(image, weight=0.05, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=500)
    g = future.graph.rag_mean_color(image, labels1)

    labels2 = future.graph.cut_threshold(labels1, g, 0.07)

    if __debug__:
        create_debug_fig(color.label2rgb(labels2, image), "Threshold Cut")

    # labels2 = future.graph.ncut(labels1, g)

    # if __debug__:
    #     create_debug_fig(color.label2rgb(labels2, image), "N Cut")

    bin_test = np.zeros((300, 400))
    for r in measure.regionprops(labels2):
        rr, cc = zip(*r.coords)
        bin_test[rr, cc] = 1

    return bin_test


# THIS IS A VERY BAD SOLUTION
def create_bin_img_threshold(image):

    img = color.rgb2gray(image)
    val = filters.threshold_adaptive(img, 35)
    result = img > val

    result = morphology.binary_opening(result, selem=morphology.disk(2))
    labels = measure.label(result)

    for region in measure.regionprops(labels):
        if region.area < 50:
            rr, cc = zip(*region.coords)
            result[rr, cc] = 0

    result = morphology.binary_closing(result, selem=morphology.disk(2))

    dist = ndi.distance_transform_edt(result)
    maxima = feature.peak_local_max(dist)

    plt.figure()
    plt.subplot(122)
    plt.imshow(result, cmap='gray')
    plt.plot(*reversed(list(zip(*maxima))), 'bo')

    dist_invert = ndi.distance_transform_edt(np.invert(result))
    minima = feature.peak_local_max(dist_invert)

    plt.plot(*reversed(list(zip(*minima))), 'ro')

    # for m in minima:
    #     r = dist_invert[m[0], m[1]]
    #     result[draw.circle(m[0], m[1], r, shape=result.shape)] = 1

    plt.subplot(121)
    plt.imshow(morphology.skeletonize(result), cmap='gray')
    
    plt.show()

    return result


def create_bin_img_edge(image, s=1):

    img = color.rgb2gray(image)

    plt.imshow(img, cmap='gray')
    plt.show()

    img *= 1 / img.max()
    bright_spots = np.array((img > 0.96).nonzero()).T
    dark_spots = np.array((img < 0.04).nonzero()).T

    plt.imshow(img, cmap='gray')
    plt.show()

    img = filters.sobel(img)
    img = filters.gaussian(img, sigma=s)

    plt.imshow(img, cmap='gray')
    plt.show()

    bool_mask = np.zeros(img.shape, dtype=np.bool)
    bool_mask[tuple(bright_spots.T)] = True
    bool_mask[tuple(dark_spots.T)] = True
    seed_mask, num_seeds = ndi.label(bool_mask)
    print(num_seeds)

    img = exposure.equalize_hist(img)
    plt.imshow(img)
    plt.show()

    ws = morphology.watershed(img, seed_mask)

    return ws