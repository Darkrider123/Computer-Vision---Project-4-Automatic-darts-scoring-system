import cv2 as cv
import numpy as np


def rotate_image(image, degrees, center="Auto", scale=1):
    height, width, _ = image.shape
    if center == "Auto":
        center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, degrees, scale)
    rotated = cv.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def translate_image(image, ox_pixels, oy_pixels):
    height, width, _ = image.shape
    affine_transformation_matrix = np.float32([[1, 0, ox_pixels],
                                            [0, 1, oy_pixels]])
    warped = cv.warpAffine(image, affine_transformation_matrix,  (width, height))
    return warped

def affine_transformation_warp(image, points_original_place_3, points_desired_place_3):
    assert len(image.shape) == 2 , f"Image must be grayscale. Image shape is not a 2 by 2, it is {image.shape}"
    height, width = image.shape
    affine_transformation_matrix = cv.getAffineTransform(points_original_place_3, points_desired_place_3)
    warped = cv.warpAffine(image, affine_transformation_matrix, (width, height))
    return warped

def perspective_transformation_warp(image, points_original_place_4, points_desired_place_4):
    assert len(image.shape) == 2 , f"Image must be grayscale. Image shape is not a 2 by 2, it is {image.shape}"
    points_original_place_4 = np.array(points_original_place_4, np.float32)
    points_desired_place_4 = np.array(points_desired_place_4, np.float32)


    height, width = image.shape
    perspective_transform_matrix = cv.getPerspectiveTransform(points_original_place_4, points_desired_place_4)
    warped = cv.warpPerspective(image, perspective_transform_matrix, (width, height))
    return warped


def resize_image(image, width, height, interpolation_method=cv.INTER_CUBIC):
    if width < 10:
        _, width, _ = image.shape * width
    if height < 10:
        height, _, _ = image.shape * height
    resized_image = cv.resize(image, (width, height), interpolation=interpolation_method)
    return resized_image