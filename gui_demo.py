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
from skimage import morphology, color, draw

from util.imgutil import draw_data_points as ddp

from PIL import Image, ImageTk
import threading
import queue

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

io.use_plugin('matplotlib')


class PhotoCanvas(Tk.Canvas):

    def __init__(self, parent, **kwargs):
        Tk.Canvas.__init__(self, parent, **kwargs)
        parent.bind("<Configure>", self.on_resize)
        self.width = self.winfo_reqwidth()
        self.height = self.winfo_reqheight()
        self.ratio = float(self.width) / self.height

    def set_image(self, image):

        self.image = image

        ratio = float(image.width) / image.height

        if ratio != self.ratio:
            if self.ratio > ratio:
                self.config(width=int(ratio * self.height), height=self.height)
            else:
                self.config(width=self.width, height=int(self.width / ratio))
            self.ratio = ratio

        self.image = self.image.resize((self.width, self.height), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(self.image)

        self.delete("all")
        self.create_image(0, 0, image=self.photo, anchor=Tk.NW)

    def on_resize(self, event):

        ratio = float(event.width) / event.height

        if ratio != self.ratio:
            if self.ratio > ratio:
                self.config(width=event.width, height=int(event.width / self.ratio))
                self.width = event.width
                self.height = int(event.width / self.ratio)
            else:
                self.config(width=int(event.height * self.ratio), height=event.height)
                self.width = int(event.height * self.ratio)
                self.height = event.height
        else:
            self.config(width=event.width, height=event.height)
            self.width = event.width
            self.height = event.height

        self.image = self.image.resize((self.width, self.height), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(self.image)

        self.delete("all")
        self.create_image(0, 0, image=self.photo, anchor=Tk.NW)


def create_window():

    w, h = 400, 300
    window = Tk.Tk()
    window.title("Hand detection")
    window.configure(background='black')

    window.overrideredirect(True)
    window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
    window.bind("<Escape>", lambda e: window.destroy())

    canvas = PhotoCanvas(window, width=w, height=h, highlightthickness=0)
    canvas.pack()

    return window, canvas


def get_photo(canvas):
    photo = Tk.PhotoImage(master=canvas, width=400, height=300)
    canvas.create_image(200, 150, image=photo)

    return photo


def identify_image(image, out_queue):

    time.sleep(1)

    def send(img, n):
        out_queue.put(img)
        time.sleep(n)

    binary, closed = segmentation.create_bin_img_otsu(image)

    send(binary, 0.25)

    closed_bin = binary
    if closed:
        closed_bin = skimage.img_as_float(morphology.binary_closing(binary, selem=morphology.disk(8)))

    palm_point, inner_radius = hd.find_palm_point(closed_bin)

    send(ddp(binary, [1, 0, 0], 2, palm_point), 0.25)

    palm_mask = hd.create_palm_mask(closed_bin, palm_point, inner_radius, 5)
    wrist_points = hd.find_wrist_points(palm_mask)

    send(ddp(ddp(binary, [0, 0, 1], 2, *palm_mask), [1, 0, 0], 2, palm_point), 0.25)


def start(path, files):

    window, canvas = create_window()
    ind = random.randrange(len(files))

    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()

    f = files[ind]
    image = img_as_float(io.imread(os.path.join(path, f)))
    image = skimage.transform.resize(image, (300, 400, 3))

    img = Image.fromarray(np.uint8(image * 255))
    canvas.set_image(img)

    q = queue.Queue()

    t = threading.Thread(target=identify_image, args=(image, q))
    t.daemon = True
    t.start()

    while True:

        if not t.is_alive():

            ind += 1
            if ind == len(files):
                ind = 0

            f = files[ind]
            image = img_as_float(io.imread(os.path.join(path, f)))
            image = skimage.transform.resize(image, (300, 400, 3))

            img = Image.fromarray(np.uint8(image * 255))
            img = img.resize((width, height), Image.NEAREST)
            canvas.set_image(img)

            t = threading.Thread(target=identify_image, args=(image, q))
            t.daemon = True
            t.start()

        try:

            item = q.get_nowait()
            img = Image.fromarray(np.uint8(item * 255))
            canvas.set_image(img)
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
