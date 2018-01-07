# imports
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage.morphology as mp
from scipy import ndimage as ndi

import os
import sys
import math
import time

import skimage
from skimage import data
from skimage import io
from skimage import img_as_float
from skimage import measure
from skimage import feature
from skimage import filters
from skimage import draw

import json

from util import imgutil
from util.imgutil import draw_data_points as ddp

from multiprocessing import Pool
from functools import partial

io.use_plugin('matplotlib')


def create_bin_img(image):
    """Create binary image from image argument and return it.

    Keyword arguments:
    image -- grayscale image to turn into binary.
    """

    # Create binary image using ohsu thresholding.
    val = filters.threshold_adaptive(image, 35)
    result = image > val

    # Check if background was brighter than foreground, if so, invert the binary image.
    labels = measure.label(result, background=2)

    # Assume that background area > foreground area
    mx_region = max(measure.regionprops(labels), key=lambda r: r.area)

    rr, cc = list(zip(*mx_region.coords))
    if np.mean(result[rr, cc]) == 1:
        result = np.invert(result)

    return result


def find_palm_point(image):
    """Find the palm point in the given binary image using a distance map.

    Keyword arguments:
    image -- binary image.
    """

    dist_map = mp.distance_transform_edt(image)
    max_index = np.argmax(dist_map)

    index = np.unravel_index(max_index, dist_map.shape)

    return index, math.ceil(dist_map[index])


def get_nearest_border(image, point):
    """Gets the closest background point in the binary image from the given point.

    Keyword arguments:
    image -- binary image.
    point -- the point to get closest point to.
    """
    radius = 1
    angle = 0

    height, width = image.shape
    for radius in range(1, image.shape[0]):
        for angle in range(0, 360, int(360 / (min(radius, 9) * 4))):

            y = math.floor(point[0] + radius * math.sin(math.radians(angle)))
            x = math.floor(point[1] + radius * math.cos(math.radians(angle)))

            if y < 0 or y >= height or x < 0 or x >= width:
                continue

            if image[y][x] == 0:

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        sy, sx = [min(height - 1, max(0, y + dy)), min(width - 1, max(0, x + dx))]
                        if image[sy][sx] != 0:
                            return [min(height - 1, max(0, y - dy)), min(width - 1, max(0, x - dx))]


def create_palm_mask(image, palm_point, inner_radius, angle_step):
    """Create a palm mask using the given binary image and palm_point.

    Keyword arguments:
    image -- binary image.
    palm_point -- the center point of the palm in the image.
    inner_radius -- the radius of the largest possible circle inside the palm.
    angle_step -- how much the angle of each point should change.
    """
    radius = inner_radius * 1.2

    angle_range = range(0, 360, angle_step)

    # Generate sample points
    sample_y = np.array([math.floor(radius * math.sin(math.radians(a)) + palm_point[0]) for a in angle_range])
    sample_x = np.array([math.floor(radius * math.cos(math.radians(a)) + palm_point[1]) for a in angle_range])
    sample_points = np.array(list(zip(sample_y, sample_x)))
    sample_points = sample_points[np.where(np.logical_and(np.logical_and(0 <= sample_y, sample_y < image.shape[0]), np.logical_and(0 <= sample_x, sample_x < image.shape[1])))]

    res = [get_nearest_border(image, p) for p in sample_points]
    return res


def find_wrist_points(palm_mask):
    """Find the wrist points in the given palm mask and calculate the middle of the wrist.

    Keyword arguments:
    palm_mask -- the palm mask to extrapolate wrist points from.
    """
    wp1 = [0, 0]
    wp2 = [0, 0]
    max_dist = 0

    prev = palm_mask[-1]
    for i in range(0, len(palm_mask)):
        p = palm_mask[i]
        dist = math.sqrt((p[0] - prev[0])**2 + (p[1] - prev[1])**2)
        if dist > max_dist:
            max_dist = dist
            wp1 = prev
            wp2 = p
        prev = p

    angle = math.atan2(wp2[0] - wp1[0], wp2[1] - wp1[1])

    mid = [math.floor(wp1[0] + math.sin(angle) * max_dist / 2), math.floor(wp1[1] + math.cos(angle) * max_dist / 2)]

    return [wp1, mid, wp2]


def rotate_hand_upright(image, palm_point, palm_mask, wrist_points):
    """Rotates the given image as well as the data points so that
    the vector between the middle wrist point and the palm point is upright.

    Keyword arguments:
    image -- binary image.
    palm_point -- the palm point.
    palm_mask -- the palm mask.
    wrist_points -- the wrist points.
    """
    _, wrist, _ = wrist_points

    angle = 90 - math.degrees(math.atan2(wrist[0] - palm_point[0], palm_point[1] - wrist[1]))

    points = {'pp': [palm_point], 'pm': palm_mask, 'wp': wrist_points}
    img_rot, points = imgutil.rotate_image(image, points, angle, True)

    palm_point_rot = points['pp'][0]
    palm_mask_rot = points['pm']
    wrist_points_rot = points['wp']

    return img_rot, palm_point_rot, palm_mask_rot, wrist_points_rot


def remove_wrist(image, wrist_points):
    """Remove the wrist from the image. This assumes the image and wrist points have been rotated with rotate_hand_upright.

    Keyword arguments:
    image -- the binary image to remove wrist from.
    wrist_points -- the points of the wrist.
    """

    p1, mid, p2 = wrist_points
    k = (p2[0] - p1[0]) / (p2[1] - p1[1])
    m = p2[0] - k * p2[1]

    img = image.copy()

    idx = np.zeros(img.shape, dtype=bool)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if y > k * x + m:
                idx[y, x] = True

    img[idx] = 0
    return img


def crop_and_resize_image(image, palm_point, palm_mask, wrist_points):
    """

    """
    regions = measure.label(image, background=0)

    hand = max(measure.regionprops(regions), key=lambda r: r.area)

    points = {'pp': [palm_point], 'pm': palm_mask, 'wp': wrist_points}

    img_ret, points_ret = imgutil.crop_image(image, points, hand.bbox)
    img_ret, points_ret = imgutil.resize_image(img_ret, points_ret, 200, 200)

    return img_ret, points_ret['pp'][0], points_ret['pm'], points_ret['wp']


def remove_palm(image, palm_mask):
    """ Removes area defined by the palm mask from the binary image.

    Keyword arguments:
    image -- binary image.
    palm_mask -- the palm mask to remove.
    """
    img = image.copy()

    r, c = zip(*palm_mask)
    rr, cc = skimage.draw.polygon(r, c, shape=img.shape)

    img[rr, cc] = 0
    return img


def find_fingers(image, palm_point):
    """

    """
    regions = measure.label(image, background=0)
    fingers = list(filter(lambda x: x.area > 30, measure.regionprops(regions)))

    finger_data = []
    has_thumb = False

    for finger in fingers:
        data = {}

        point = finger.centroid

        thumb = False

        _, minx, _, maxx = finger.bbox
        width = maxx - minx

        orientation = finger.orientation
        if orientation < 0:
            orientation += math.pi

        ang = abs(math.degrees(math.atan2(palm_point[0] - point[0], palm_point[1] - point[1])))
        if ang < 50:
            has_thumb = True
            thumb = True

        minor_axis = finger.minor_axis_length
        major_axis = finger.major_axis_length
        if not thumb and (math.degrees(orientation) < 30 or math.degrees(orientation) > 150):
            minor_axis = finger.major_axis_length
            major_axis = finger.minor_axis_length
            orientation -= math.pi / 2
            if orientation < 0:
                orientation += math.pi

        dy = (math.sin(orientation) * 0.5 * major_axis)
        dx = (math.cos(orientation) * 0.5 * major_axis)

        if math.floor(minor_axis / 25) > 1:
            fingers = math.floor(minor_axis / 25)        
            for x in range(0, fingers):
                dist = (-1 * minor_axis / 2) + minor_axis / (2 * fingers) + x * minor_axis / fingers

                p = [point[0] + math.cos(orientation) * dist, point[1] + math.sin(orientation) * dist]
                ang = abs(math.degrees(math.atan2(palm_point[0] - p[0], palm_point[1] - p[1])))

                d = {
                    'angle_from_pp': ang,
                    'region': finger,
                    'point': p,
                    'label': 'NO_DEF',
                    'tip': [p[0] - dy, p[1] + dx],
                    'start': [p[0] + dy, p[1] - dx]
                }
                finger_data.append(d)
        else:

            data['angle_from_pp'] = ang
            data['region'] = finger
            data['point'] = finger.centroid
            data['label'] = "NO_DEF"
            data['tip'] = [point[0] - dy, point[1] + dx]
            data['start'] = [point[0] + dy, point[1] - dx]

            if thumb:
                finger_data.insert(0, data)
            else:
                finger_data.append(data)

    return finger_data, has_thumb


def get_nearest_finger(pos, finger_data):
    """

    """
    def distance(p):

        dy = pos[0] - p[1]['point'][0]
        dx = pos[1] - p[1]['point'][1]

        return math.sqrt(dy**2 + dx**2)

    return min(enumerate(finger_data), key=distance)[0]


def find_palm_line(image, wrist_points, finger_data, has_thumb):
    """

    """
    pline = -1
    line = int(wrist_points[1][0])
    start = 0
    end = 0

    while pline == -1:
        w = False
        occ = 0

        for i in range(0, image.shape[1]):
            if image[line, i] == 1:
                if not w:
                    w = True
                    if occ == 0:
                        start = i

            else:
                if w:
                    if not has_thumb or get_nearest_finger([line, i], finger_data) != 0:
                        occ += 1
                    w = False
                if occ == 2:
                    pline = line
                    end = i
                    break

        line = line - 1

    result = {}
    result['line'] = pline
    result['start'] = start
    result['end'] = end

    return result


def identify_fingers(finger_data, palm_line, has_thumb):
    """

    """
    for idx, finger in enumerate(finger_data):
        if idx == 0 and has_thumb:
            finger['label'] = 'thumb'
            continue

        x = finger['point'][1]

        interval = (palm_line['end'] - palm_line['start']) / 4
        start = palm_line['start']

        if x < start + interval:
            finger['label'] = 'index'
        elif x < start + 2 * interval:
            finger['label'] = 'middle'
        elif x < start + 3 * interval:
            finger['label'] = 'ring'
        else:
            finger['label'] = 'pinky'

    return


def write_data(finger_data, palm_point, file_name, directory=''):

    data = {}
    data['fingers'] = finger_data
    data['palm_point'] = palm_point

    with open(directory + file_name, 'w+') as outfile:
        json.dump(data, outfile)


def draw_finished_image(image, finger_data, palm_point):
    canvas = skimage.color.gray2rgb(image.copy())

    colors = {
        'thumb': [1, 0, 0],
        'index': [0, 1, 0],
        'middle': [0, 0, 1],
        'ring': [1, 1, 0],
        'pinky': [1, 0.5, 0.5],
        'NO_DEF': [0.5, 0.5, 0.5]
    }

    for finger in finger_data:
        c = colors[finger['label']]
        canvas[draw.circle(finger['point'][0], finger['point'][1], 3, shape=canvas.shape)] = c
        canvas[draw.circle(finger['start'][0], finger['start'][1], 3, shape=canvas.shape)] = c
        canvas[draw.circle(finger['tip'][0], finger['tip'][1], 3, shape=canvas.shape)] = [1, 0, 0]

        canvas[draw.line(int(palm_point[0]), int(palm_point[1]), int(finger['start'][0]), int(finger['start'][1]))] = c
        canvas[draw.line(int(finger['start'][0]), int(finger['start'][1]), int(finger['point'][0]), int(finger['point'][1]))] = c
        canvas[draw.line(int(finger['point'][0]), int(finger['point'][1]), int(finger['tip'][0]), int(finger['tip'][1]))] = c

    canvas[draw.circle(palm_point[0], palm_point[1], 5, shape=canvas.shape)] = [0, 0, 0]
    return canvas


def identify_image(image_path, outdir):

    name = os.path.split(image_path)[1]
    print("Processing image:", image_path)
    image = io.imread(image_path)
    image = img_as_float(image)
    image = skimage.transform.resize(image, (300, 400, 3))

    image = skimage.color.rgb2gray(image)

    binary = create_bin_img(image)

    plt.imshow(binary, cmap='gray')
    plt.show()

    palm_point, inner_radius = find_palm_point(binary)

    palm_mask = create_palm_mask(binary, palm_point, inner_radius, 5)
    wrist_points = find_wrist_points(palm_mask)

    binary, palm_point, palm_mask, wrist_points = rotate_hand_upright(binary, palm_point, palm_mask, wrist_points)
    no_wrist_img = remove_wrist(binary, wrist_points)

    no_wrist_img, palm_point, palm_mask, wrist_points = crop_and_resize_image(no_wrist_img, palm_point, palm_mask, wrist_points)

    no_palm_img = remove_palm(no_wrist_img, palm_mask)
    finger_data, has_thumb = find_fingers(no_palm_img, palm_point)

    palm_line = find_palm_line(no_wrist_img, wrist_points, finger_data, has_thumb)

    identify_fingers(finger_data, palm_line, has_thumb)

    for finger in finger_data:
        del finger['region']

    drawing = draw_finished_image(no_wrist_img, finger_data, palm_point)

    print("Sucessfully processed:", os.path.split(image_path)[1])
    return drawing, finger_data, palm_point


def identify_and_output(image_path, outdir):
    image, finger_data, palm_point = identify_image(image_path, outdir)
    write_data(finger_data, palm_point, os.path.splitext(os.path.split(image_path)[1])[0] + '_data.json', outdir)
    io.imsave(os.path.join(outdir, os.path.splitext(os.path.split(image_path)[1])[0] + '_image.jpg'), image)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Usage: python', sys.argv[0], 'path-to-img|path-to-dir', '[out dir]')
    else:
        outdir = ''
        if len(sys.argv) == 3:
            outdir = sys.argv[2]
            if outdir[-1] != '/':
                outdir += '/'
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        path = sys.argv[1]

        if os.path.isfile(path):
            identify_and_output(path, outdir)
        elif os.path.isdir(path):

            files = []
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path, f)):
                    files.append(os.path.join(path, f))

            p = Pool()
            p.map(partial(identify_and_output, outdir=outdir), files)
            # for f in files:
            #    identify_and_output(f, outdir)

            print("All images processed!")
        else:
            print("Path specified is not a valid file or directory:", path)
