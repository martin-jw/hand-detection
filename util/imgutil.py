"""Module containing helpers functions for numpy images with related data points.
"""

from skimage import transform
from skimage import draw
from skimage import color

import math

def resize_image(image, points, width, height):
    """Resize the given image and translate the given data points accordingly.

    Resizes the given numpy array image to the given width and height. The offsets the
    given data points to correspond to the correct location of the new resized image.
    Points are expected to be a python dictionary where the values are arrays of
    shape (n, 2) (arrays of 2D points).

    Keyword arguments:
    image -- a numpy array representing an image.
    points -- a dict of arrays with points to translate.
    width -- the width to resize to
    height -- the height to resize to
    """
    s = list(image.shape)
    s[0] = height
    s[1] = width
    s = tuple(s)
    image_res = transform.resize(image, s)
    shape = image.shape

    def move_point(p):
        return [(p[0] / shape[0]) * height, (p[1] / shape[1]) * width]

    points_res = {k: list(map(move_point, v)) for k, v in points.items()}

    return image_res, points_res


def crop_image(image, points, bbox):
    """Crop the image and translate the given data points to the new coordinates.

    Crops the given numpy array image to the given bounding box. Then offsets the
    given data points to correspond to the correct location of the new cropped image. Points are expected
    to be a python dictionary where the values are arrays of shape (n, 2) (arrays of 2D points).

    Keyword arguments:
    image -- a numpy array representing an image.
    points -- a dict of arrays with points to translate.
    bbox -- the bounding box to crop to.
    """
    miny, minx, maxy, maxx = bbox

    miny -= 5
    minx -= 5
    maxy += 5
    maxx += 5

    img = image.copy()
    img = img[miny:maxy, minx:maxx]

    def move_point(point):
        return [point[0] - miny, point[1] - minx]

    points_res = {k: list(map(move_point, v)) for k, v in points.items()}

    return img, points_res


def rotate_image(image, points, angle, resize=True):
    """Rotate the image and translate the given data points accordingly.

    Rotates the given numpy array image to the angle. Then offsets the
    given data points to correspond to the correct location of the new rotated image.
    Points are expected to be a python dictionary where the values are arrays of
    shape (n, 2) (arrays of 2D points).

    Keyword arguments:
    image -- a numpy array image.
    points -- a dict of arrays with points to translate.
    angle -- the angle with which to rotate (in degrees).
    resize -- wether to resize the image to fit the entire rotation or not (default: True)
    """

    cy = image.shape[0] / 2
    cx = image.shape[1] / 2
    img_rot = transform.rotate(image, angle, resize)

    oy = img_rot.shape[0] / 2 - cy
    ox = img_rot.shape[1] / 2 - cx

    angle = math.radians(angle)

    tform = transform.SimilarityTransform(scale=1, rotation=angle, translation=(oy, ox))

    def move_point(p):
        dy = p[0] - cy
        dx = p[1] - cx
        po = tform([dy, dx])[0]
        return [cy + po[0], cx + po[1]]

    points_res = {k: list(map(move_point, v)) for k, v in points.items()}

    return img_rot, points_res


def draw_data_points(image, col, radius, *args):
    """Draw the given data points on the image using the color specified.

    Keyword arguments:
    image -- a numpy array image.
    col -- the color with which to draw.
    radius -- the radius with which the points are drawn.
    *args -- any number of arrays (points) with len(point) == 2
    """
    img = image.copy()

    if img.ndim == 2:
        img = color.gray2rgb(img)

    assert img.shape[-1] == len(col)

    for point in args:
        if len(point) != 2:
            continue

        img[draw.circle(point[0], point[1], radius, shape=img.shape)] = col

    return img
