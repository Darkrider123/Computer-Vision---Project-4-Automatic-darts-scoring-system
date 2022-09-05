import cv2 as cv
import numpy as np

def filter_image(image, kernel):
    filtered_image = cv.filter2D(image, -1, kernel)
    return filtered_image

def smoothen_image_gaussian_filter(image, kernel_size, sigmaX=0):
    if kernel_size % 2 == 0:
        raise ValueError("kernel size should be odd")

    kernel_size = (kernel_size, kernel_size)
    return cv.GaussianBlur(image, kernel_size, sigmaX)

def smoothen_image_bilateral_filter(image):
    return cv.bilateralFilter(image, -1, 300, 20)

def sharpen_image(image, ammount):
    kernel = np.array([[-1, -1, -1],
                     [-1, ammount, -1],
                     [-1, -1, -1]])
    return filter_image(image, kernel)


def remove_noise_from_image_median_filter(image, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("kernel size should be odd")
    return cv.medianBlur(image, kernel_size)