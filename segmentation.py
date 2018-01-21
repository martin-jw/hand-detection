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

from scipy import ndimage as ndi

import matplotlib.pyplot as plt


def create_bin_img_slic(image):

    image = restoration.denoise_tv_chambolle(image, weight=0.05, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=500)
    g = future.graph.rag_mean_color(image, labels1)

    plt.subplot(121)
    plt.imshow(color.label2rgb(labels1, image))

    labels2 = future.graph.cut_threshold(labels1, g, 0.060)
    # labels2 = future.graph.ncut(labels1, g)

    plt.subplot(122)
    plt.imshow(color.label2rgb(labels2, image))

    # plt.show()

    edges = feature.canny(color.rgb2gray(image), sigma=1.5)
    plt.imshow(edges, cmap='gray')
    # plt.show()

    bin_test = np.zeros((300, 400))
    for r in measure.regionprops(labels2):
        rr, cc = zip(*r.coords)
        bin_test[rr, cc] = 1

    return bin_test


# THIS IS A VERY BAD SOLUTION
def create_bin_img_threshold(image):

    img = color.rgb2gray(image)
    val = filters.threshold_local(img, 35)
    result = img > val

    mask = create_bin_img_slic(img)

    # Check if background was brighter than foreground, if so, invert the binary image.
    labels = measure.label(result, background=2, connectivity=1)

    regions = measure.regionprops(labels)
    # Assume that background area > foreground area
    mx_region = max(regions, key=lambda r: r.area)
    regions.remove(mx_region)

    for r in regions:
        if r.area < 400:
            rr, cc = zip(*r.coords)
            result[rr, cc] = 0

    rr, cc = list(zip(*mx_region.coords))
    if np.mean(result[rr, cc]) == 1:
        result = np.invert(result)

    mask_invert = np.ones(mask.shape, dtype="bool")
    mask_invert[np.where(mask)] = False

    result[np.where(mask_invert)] = 0
    result = morphology.binary_closing(result, morphology.disk(5))

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