import matplotlib.pyplot as plt
import cv2 as cv
import os


def load_image(path, filename):
    return plt.imread(os.path.join(path, filename))

def load_image_cv(path, filename, grayscale=False):
    if grayscale is True:
        image = cv.imread(os.path.join(path, filename), cv.IMREAD_GRAYSCALE)
    else:
        image = cv.imread(os.path.join(path, filename))

    return cv.cvtColor(image, cv.COLOR_BGR2RGB)