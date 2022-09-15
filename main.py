import numpy as np
import os

from graphic_display import draw_circle, draw_line, show_image
from feature_extraction_methodes.mid_level_feature_extraction_methodes import hough, color_segmentation
from feature_extraction_methodes.low_level_feature_extraction_methodes import edge_extraction
from image_processing.colors import *
from image_processing.color_processing import get_grayscale_image, get_color_from_grayscale_image
from image_processing.image_manipulation import perspective_transformation_with_4_points, resize_image, pad_image
from image_processing.filters import sharpen_image, smoothen_image_gaussian_filter, remove_noise_from_image_median_filter
from data_manipulation.data_loader import load_image_cv
from data_manipulation.data_dumper import save_image
from image_processing.image_stitching import stitch_image_inside


def main():
    task1()
    task2()


def task1():
    image_folder_path = os.path.join("project_files", "train", "Task1")
    files = os.listdir(image_folder_path)
    files = [file for file in files if file.split(".")[1]=="jpg"]

    for file in files:
        image = load_image_cv(image_folder_path, file)
        process_image_task1(image)

def task2():
    image_folder_path = os.path.join("project_files", "train", "Task2")
    files = os.listdir(image_folder_path)
    files = [file for file in files if file.split(".")[1]=="jpg"]

    for file in enumerate(files):
        image = load_image_cv(image_folder_path, file)
        process_image_task2(image)


def change_perspective_of_target(image):
    points_original = [[103, 455], [516, 104], [815, 434], [542, 783]]
    points_desired = [[100, 500], [500, 100], [900, 500], [500, 900]]
    image = perspective_transformation_with_4_points(image, points_original, points_desired)
    return image


def process_image_hough_circles(image):
    image = resize_image(image, 1000, 1000)
    #image, segments_color = color_segmentation.segment_image_based_on_color_KMeans(image, 2)
    image = get_grayscale_image(image)
    image = remove_noise_from_image_median_filter(image, 3)
    image = smoothen_image_gaussian_filter(image, 7, 1.5)
    image = change_perspective_of_target(image)
    #image = edge_extraction.Canny(image, 300, 290, 5)

    #show_image(image)
#
    #lines = hough.get_lines_from_image(image, 150, 10, 10)
    #for line in lines:
    #    draw_line(image, (line[0], line[1]), (line[2], line[3]), ORANGE)
    #show_image(image, True)
    #exit()

    circles = hough.get_circles_from_image(image, 0, int(np.mean([image.shape[0], image.shape[1]])), 40, 200, 60, 1, 100)
    image = get_color_from_grayscale_image(image)

    if circles is not None:
        for circle in circles:
            draw_circle(image, circle, MAGENTA, 3)
            draw_circle(image, (circle[0], circle[1], 2), BLUE, -1)
    
        print(len(circles))
    show_image(image, True)



def contours(image):
    image, segments_color = color_segmentation.segment_image_based_on_color_KMeans(image, 2)
    show_image(image, True)
    image = get_grayscale_image(image)
    image = remove_noise_from_image_median_filter(image, 11)
    image = smoothen_image_gaussian_filter(image, 7, 1.5)
    image = edge_extraction.Canny(image, 300, 290, 5)


    import cv2 as cv
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = get_color_from_grayscale_image(image)
    image = cv.drawContours(image, contours, -1, BLUE, 10)
    show_image(image, True)



def process_image_task1(image):
    pass

def process_image_task2(image):
    pass


if __name__ == '__main__':
    pass
    