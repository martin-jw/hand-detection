"""

TURN AROUND WHILE YOU STILL CAN!

This code is undocumented and poorly written, don't look at it.
I wrote it in like 2 days.
"""
import matplotlib as mpl
import numpy as np
import sys
import os
import time
import random

import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

import hand_detect as hd
import segmentation

import skimage
from skimage import io
from skimage import img_as_float
from skimage import morphology, color, draw, restoration
from skimage import segmentation, filters, measure, future
from skimage.filters import rank

from scipy import ndimage as ndi

from util.imgutil import draw_data_points as ddp

from PIL import Image, ImageTk
import threading
import queue
from functools import partial

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

io.use_plugin('matplotlib')


def create_bin_img_slic(image, send_func, segments=500, thresh=0.1):
    """A background segmentation algorithm. Segments the image using Simple Linear Iterative Clustering and a graph cut. One
    region is chosen as the background and the rest is chosen as foreground.

    Keyword arguments:
    image -- The image to segment.
    segments -- The number of segments SLIC should segment into. Default is 500.
    thresh -- The threshold for the graph cut. Default is 0.1."""

    image = restoration.denoise_tv_chambolle(image, weight=0.04, multichannel=True)

    labels1 = segmentation.slic(image, n_segments=segments)
    g = future.graph.rag_mean_color(image, labels1)

    send_func(("img", color.label2rgb(labels1, image)), 0.5)

    labels2 = future.graph.cut_threshold(labels1, g, thresh)
    send_func(("img", color.label2rgb(labels2, image)), 0.5)

    bin_test = np.zeros((300, 400))
    for r in measure.regionprops(labels2):
        rr, cc = zip(*r.coords)
        bin_test[rr, cc] = 1

    return bin_test, False


def create_bin_img_otsu(image, send_func):

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


def create_bin_img_watershed(image, send_func):

    img = color.rgb2gray(image)
    denoised = rank.median(img, morphology.disk(1))

    markers = rank.gradient(denoised, morphology.disk(4)) < 20
    markers = ndi.label(markers)[0]
    send_func(("img", markers), 0.5)

    gradient = rank.gradient(denoised, morphology.disk(2))
    send_func(("img", gradient), 0.5)

    labels = morphology.watershed(gradient, markers)
    send_func(("img", labels), 0.5)
    binary = np.zeros((labels.shape[0], labels.shape[1]), dtype='float64')

    for r in measure.regionprops(labels):
        if r.label != 1:
            rr, cc = zip(*r.coords)
            binary[rr, cc] = 1

    return binary, False


class PhotoCanvas(Tk.Canvas):

    def __init__(self, parent, **kwargs):
        Tk.Canvas.__init__(self, parent, **kwargs)
        self.target_width = self.winfo_reqwidth()
        self.target_height = self.winfo_reqheight()
        self.width = self.target_width
        self.height = self.target_height

    def set_image(self, image):

        self.image = image

        ratio = float(image.width) / image.height

        new_w = self.target_width
        new_h = int(self.target_width / ratio)

        if (new_h > self.target_height):
            new_w = int(ratio * self.target_height)
            new_h = self.target_height

        self.config(width=new_w, height=new_h)
        self.width = new_w
        self.height = new_h

        self.image = self.image.resize((self.width, self.height), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(self.image)

        self.delete("image")
        self.create_image(0, 0, image=self.photo, anchor=Tk.NW, tags="image")
        self.tag_raise("text")

    def set_text(self, text):

        self.delete("text")
        self.create_text((10, 10), anchor=Tk.NW, text=text, fill="white", font=("Purisa", 18), tags="text")

    def add_point(self, p, img_size, color):

        r = 4
        y = int(self.height * (float(p[0]) / img_size[0]))
        x = int(self.width * (float(p[1]) / img_size[1]))

        self.create_oval((x - r, y - r, x + r, y + r), fill=color, tags="image")


paused = False
delayMod = 1


def toggle_pause(e):
    global paused
    paused = not paused


def inc_delay(e):
    global delayMod
    delayMod += 1.2
    if delayMod > 3:
        delayMod = 3


def dec_delay(e):
    global delayMod
    delayMod -= 0.2
    if delayMod < 0.2:
        delayMod = 0.2


def create_window():

    window = Tk.Tk()
    window.title("Hand detection")
    window.configure(background='black')

    w, h = window.winfo_screenwidth(), window.winfo_screenheight()

    window.overrideredirect(True)
    window.geometry("{0}x{1}+0+0".format(w, h))
    window.bind("<Escape>", lambda e: window.destroy())
    window.bind("<space>", toggle_pause)
    window.bind("<Left>", inc_delay)
    window.bind("<Right>", dec_delay)

    window.resizable(False, False)

    canvas = PhotoCanvas(window, width=w, height=h, highlightthickness=0)
    canvas.pack()

    return window, canvas


def get_photo(canvas):
    photo = Tk.PhotoImage(master=canvas, width=400, height=300)
    canvas.create_image(200, 150, image=photo)

    return photo


def identify_image(image, create_bin_func, out_queue):
    global delayMod

    time.sleep(1 * delayMod)

    def send(img, n):
        while paused:
            time.sleep(0.1)
        out_queue.put(img)
        time.sleep(n * delayMod)

    binary, closed = create_bin_func(image, send)

    send(("img", binary), 0.5)

    closed_bin = binary
    if closed:
        closed_bin = skimage.img_as_float(morphology.binary_closing(binary, selem=morphology.disk(8)))

    palm_point, inner_radius = hd.find_palm_point(closed_bin)

    send(("img", binary), 0)
    send(("p", {"points": [palm_point], "img_shape": binary.shape, "color": "red"}), 0.5)

    palm_mask = hd.create_palm_mask(closed_bin, palm_point, inner_radius, 5)
    wrist_points = hd.find_wrist_points(palm_mask)

    send(("img", binary), 0)
    send(("p", {"points": palm_mask, "img_shape": binary.shape, "color": "blue"}), 0)
    send(("p", {"points": [palm_point], "img_shape": binary.shape, "color": "red"}), 0.5)

    binary, palm_point, palm_mask, wrist_points = hd.rotate_hand_upright(binary, palm_point, palm_mask, wrist_points)
    send(("img", binary), 0.5)
    send(("p", {"points": palm_mask, "img_shape": binary.shape, "color": "blue"}), 0)
    send(("p", {"points": [palm_point], "img_shape": binary.shape, "color": "red"}), 0.5)
    send(("p", {"points": wrist_points, "img_shape": binary.shape, "color": "purple"}), 0.5)

    no_wrist_img = hd.remove_wrist(binary, wrist_points)
    send(("img", no_wrist_img), 0)
    send(("p", {"points": wrist_points, "img_shape": no_wrist_img.shape, "color": "purple"}), 0.5)

    no_wrist_img, palm_point, palm_mask, wrist_points = hd.crop_and_resize_image(no_wrist_img, palm_point, palm_mask, wrist_points)
    send(("img", no_wrist_img), 0)
    send(("p", {"points": palm_mask, "img_shape": no_wrist_img.shape, "color": "blue"}), 0)
    send(("p", {"points": [palm_point], "img_shape": no_wrist_img.shape, "color": "red"}), 0)
    send(("p", {"points": wrist_points, "img_shape": no_wrist_img.shape, "color": "purple"}), 0.5)

    no_palm_img = hd.remove_palm(no_wrist_img, palm_mask)
    finger_data, has_thumb = hd.find_fingers(no_palm_img, palm_point)

    send(("img", no_palm_img), 0)
    send(("p", {"points": palm_mask, "img_shape": no_wrist_img.shape, "color": "blue"}), 0)
    send(("p", {"points": [palm_point], "img_shape": no_wrist_img.shape, "color": "red"}), 0.5)

    closed_no_wrist = no_wrist_img

    if closed:
        closed_no_wrist = skimage.img_as_float(morphology.binary_closing(no_wrist_img, selem=morphology.disk(8)))

    palm_line = hd.find_palm_line(closed_no_wrist, wrist_points, finger_data, has_thumb)

    hd.identify_fingers(finger_data, palm_point, palm_line, has_thumb)

    for finger in finger_data:
        del finger['region']

    drawing = hd.draw_finished_image(no_wrist_img, finger_data, palm_point)
    send(("img", drawing), 1)


def handle_tag(tag, obj, canvas):

    if tag == "img":
        img = Image.fromarray(np.uint8(obj * 255))
        canvas.set_image(img)

    elif tag == "p":

        for p in obj["points"]:
            canvas.add_point(p, obj["img_shape"], obj["color"])


def start(path, files):

    window, canvas = create_window()
    ind = random.randrange(len(files))
    func_ind = 0

    functions = [create_bin_img_otsu, create_bin_img_watershed,
                 create_bin_img_slic, partial(create_bin_img_slic, thresh=0.045)]

    func_names = ["Otsu's method", "Watershed", "SLIC t=0.1", "SLIC t=0.045"]

    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()

    f = files[ind]
    image = img_as_float(io.imread(os.path.join(path, f)))
    image_res = skimage.transform.resize(image, (300, 400, 3))

    img = Image.fromarray(np.uint8(image * 255))
    canvas.set_image(img)
    canvas.set_text("{0} - {1}".format(f, func_names[func_ind]))

    q = queue.Queue()

    t = threading.Thread(target=identify_image, args=(image_res, functions[func_ind], q))
    t.daemon = True
    t.start()

    while True:

        if not t.is_alive():

            func_ind += 1

            if func_ind == len(functions):
                func_ind = 0
                ind = random.randrange(len(files))

                f = files[ind]
                image = img_as_float(io.imread(os.path.join(path, f)))
                image_res = skimage.transform.resize(image, (300, 400, 3))

            img = Image.fromarray(np.uint8(image * 255))
            img = img.resize((width, height), Image.NEAREST)
            canvas.set_image(img)
            canvas.set_text("{0} - {1}".format(f, func_names[func_ind]))

            t = threading.Thread(target=identify_image, args=(image_res, functions[func_ind], q))
            t.daemon = True
            t.start()

        try:

            item = q.get_nowait()
            tag, obj = item

            handle_tag(tag, obj, canvas)

        except queue.Empty:
            pass

        window.update_idletasks()
        window.update()

        time.sleep(0.01)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python", sys.argv[0], "path-to-dir")
        sys.exit()

    dir_path = sys.argv[1]
    files = []

    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, f)):
                files.append(f)

    start(dir_path, files)
