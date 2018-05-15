import matplotlib.pyplot as plt

debug_fig = None
subplot_index = 1


def create_debug_fig(img, title, cmap=None):
    """Creates a new subplot on the debug figure for the given image and title.

    Keyword arguments:
    img -- The image to show on the subplot.
    title -- The title to give to the subplot.
    cmap -- The color map to use for the subplot.
    """
    global debug_fig, subplot_index

    if debug_fig is None:
        debug_fig = plt.figure()
    ax = debug_fig.add_subplot(3, 4, subplot_index)
    ax.axis('off')

    subplot_index += 1
    if subplot_index > 12:
        subplot_index = 1

    ax.set_title(title)
    ax.imshow(img, cmap=cmap)

    return ax


def show():
    """Shows the debug figure and resets it.
    """
    global debug_fig, subplot_index

    plt.show()
    debug_fig = None
    subplot_index = 1
