import cv2 as cv

def get_grayscale_image(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

def get_color_from_grayscale_image(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2RGB)