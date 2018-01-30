import numpy as np 
from scipy import misc
import sys


def find_closest_palette_color(pix, mode):
    new_r = round(mode * pix[0] / 255) * round(255 / mode)
    new_g = round(mode * pix[1] / 255) * round(255 / mode)
    new_b = round(mode * pix[2] / 255) * round(255 / mode)
    return np.array([new_r, new_g, new_b])


def apply_filter(image, f):
    if image.ndim == 2:
        image = np.dstack((image, image, image))

    w, h, c = image.shape

    output = np.zeros((w, h))

    f_w, f_h = f.shape
    for i in range(2, w - 2):
        for j in range(3, h - 3):
            for k in range(c):
                output[i, j] += np.sum(image[i - 1:i + 2, j - 1:j + 2, k] * f)

    return output


def normalize(image, treshold):
    output = (image - image.min()) / (image.max() - image.min())
    w, h = output.shape
    for i in range(w):
        for j in range(h):
            if output[i, j] > treshold:
                output[i, j] = 1
            else:
                output[i, j] = 0

    return output


img = misc.imread(sys.argv[1])

filtr = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]])
filtr2 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

out = apply_filter(img, filtr)
out = apply_filter(out, filtr2)
out = normalize(out, 0.5)

misc.imshow(out)
misc.imsave('res.jpg', out)

