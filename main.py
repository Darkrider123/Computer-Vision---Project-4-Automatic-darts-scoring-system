from re import T
import cv2 as cv
import os

from graphic_display import draw_circle, show_image
from low_level_feature_extraction_methodes import hough
from image_processing.colors import BLUE, RED, GREEN, ORANGE
from image_processing.color_processing import get_grayscale_image, get_color_from_grayscale_image
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


def coco():
    image = load_image_cv("data", "cialapa.png")
    grayscaled_image = get_grayscale_image(image)
    circles = hough.get_circles_from_image(grayscaled_image, 0, 350, 5)

    for circle in circles:
        x, y, radius = circle
        draw_circle(image, circle, GREEN)
        draw_circle(image, (x, y, 1), BLUE, -1)
    show_image(image)





if __name__ == '__main__':
    coco()
