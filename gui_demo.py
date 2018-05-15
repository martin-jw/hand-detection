import matplotlib as mpl
import numpy as np 
import sys
import os
import time

if sys.version_info[0] < 3:
	import Tkinter as Tk 
else:
	import tkinter as Tk

import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

import hand_detect as hd

import skimage
from skimage import io
from skimage import img_as_float

from PIL import Image, ImageTk

io.use_plugin('matplotlib')


def draw_figure(canvas, figure, loc=(0, 0)):
	figure_canvas_agg = FigureCanvasAgg(figure)
	figure_canvas_agg.draw()
	figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
	figure_w, figure_h = int(figure_w), int(figure_h)
	photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

	canvas.create_image(loc[0], loc[1], image=photo)

	tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

	return photo


def create_window():

	w, h = 400, 300
	window = Tk.Tk()
	window.title("Hand detection")

	canvas = Tk.Canvas(window, width = w, height = h)
	canvas.pack()

	return window, canvas


def get_photo(canvas):
	photo = Tk.PhotoImage(master=canvas, width=400, height=300)
	canvas.create_image(200, 150, image=photo)

	return photo


def start(path, files):

	window, canvas = create_window()
	ind = 0
	t0 = 0

	while True:

		t = time.time()

		if (t - t0) > 1:
			f = files[ind]
			print(f)

			image = img_as_float(io.imread(os.path.join(path, f)))
			image = skimage.transform.resize(image, (300, 400, 3))

			img = Image.fromarray(np.uint8(image*255))
			photo = ImageTk.PhotoImage(image=img)
			canvas.create_image(0,0,image=photo,anchor=Tk.NW)

			window.update_idletasks()
			window.update()
			t0 = time.time()

			ind += 1

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
