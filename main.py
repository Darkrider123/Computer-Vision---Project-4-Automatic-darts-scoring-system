import numpy as np
import cv2 as cv
import os

from graphic_display import draw_circle, draw_line, show_image
from mid_level_feature_extraction_methodes import hough, color_segmentation
from image_processing.colors import *
from image_processing.color_processing import get_grayscale_image, get_color_from_grayscale_image
from image_processing.image_manipulation import perspective_transformation_warp, resize_image
from image_processing.filters import smoothen_image_gaussian_filter
from data_manipulation.data_loader import load_image_cv

def main():
    image_folder_path = os.path.join("data", "train", "Task2")
    files = os.listdir(image_folder_path)
    files = [file for file in files if file.split(".")[1]=="jpg"]

    for file in files:

        image = load_image_cv(image_folder_path, file)
        grayscaled_image = get_grayscale_image(image)

        circles = hough.get_circles_from_image(grayscaled_image)
        for circle in circles:
            draw_circle(image, circle, GREEN)
        show_image(image, maximize=True)


def change_perspective_of_target():
    image = load_image_cv("data", "template_task1.jpg")
    image = color_segmentation.segment_image_based_on_color_KMeans(image, 2)
    image = get_grayscale_image(image)
    points_original = [[252, 1504], [1206, 339], [1992, 1425], [1303, 2529]]
    points_desired = [[400, 1600], [1200, 800], [2000, 1600], [1200, 2400]]
    image = perspective_transformation_warp(image, points_original, points_desired)
    image = resize_image(image, 400, 400)
    circles = hough.get_circles_from_image(image, 0, 300, 17, 150, 45, 1, 3)
    image = get_color_from_grayscale_image(image)

    #for circle in circles:
    #    draw_circle(image, circle, MAGENTA, 3)
    #    draw_circle(image, (circle[0], circle[1], 2), BLUE, -1)
    
    #print(len(circles))
    show_image(image, True)



if __name__ == '__main__':
    change_perspective_of_target()