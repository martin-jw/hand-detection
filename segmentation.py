import numpy as np

from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage import future
from skimage import restoration
from skimage import morphology
from skimage import exposure
from skimage import measure


def create_bin_img_slic(image):

    image = restoration.denoise_tv_chambolle(image, weight=0.05, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=500)
    g = future.graph.rag_mean_color(image, labels1)

    labels2 = future.graph.cut_threshold(labels1, g, 0.052)
    # labels2 = future.graph.ncut(labels1, g)

    bin_test = np.zeros((300, 400))
    for r in measure.regionprops(labels2):
        rr, cc = zip(*r.coords)
        bin_test[rr, cc] = 1

    return bin_test


def create_bin_img_threshold(image):

    val = filters.threshold_local(image, 35)
    result = image > val

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
