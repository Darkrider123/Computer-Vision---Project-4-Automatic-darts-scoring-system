import numpy as np
import os

from graphic_display import draw_circle, draw_line, show_image
from feature_extraction_methodes.mid_level_feature_extraction_methodes import hough, color_segmentation, sift
from feature_extraction_methodes.low_level_feature_extraction_methodes import edge_extraction
from image_processing.colors import *
from image_processing.color_processing import get_grayscale_image, get_color_from_grayscale_image
from image_processing.image_manipulation import perspective_transformation_with_4_points, resize_image, pad_image, border_box_image
from image_processing.filters import sharpen_image, smoothen_image_gaussian_filter, remove_noise_from_image_median_filter, dilate_image
from data_manipulation.data_loader import load_image_cv, load_video_cv
from data_manipulation.data_dumper import save_image
from image_processing.image_stitching import stitch_image_inside
from image_processing.background_extraction import extract_foreground_from_image
import cv2 as cv



def main():
    submission_folder_path = "Ariton_Cosmin_406"
    #task1(submission_folder_path)
    #task2(submission_folder_path)
    task3(submission_folder_path)




def task1(submission_folder_path):
    task_folder = "Task1"
    image_folder_path = os.path.join("train", task_folder)
    files = os.listdir(image_folder_path)
    files = [file for file in files if file.split(".")[1]=="jpg"]

    for file in files:
        image = load_image_cv(image_folder_path, file)
        results = process_image_task1(image)
        
        result_string = ""
        for result in results:
            result_string = result_string + str(result) + "\n"
        result_string = result_string[:-1]

        with open(os.path.join(submission_folder_path, task_folder, file.split(".")[0] + "_predicted.txt"), "w") as f:
            f.write(result_string)





def task2(submission_folder_path):
    task_folder = "Task2"
    image_folder_path = os.path.join("train", task_folder)
    files = os.listdir(image_folder_path)
    files = [file for file in files if file.split(".")[1]=="jpg"]

    result_string = ""
    for file in files:
        image = load_image_cv(image_folder_path, file)
        results = process_image_task2(image)
        
        result_string = ""
        for result in results:
            result_string = result_string + str(result) + "\n"
        result_string = result_string[:-1]

        with open(os.path.join(submission_folder_path, task_folder, file.split(".")[0] + "_predicted.txt"), "w") as f:
            f.write(result_string)



def task3(submission_folder_path):
    task_folder = "Task3"
    image_folder_path = os.path.join("train", task_folder)
    files = os.listdir(image_folder_path)
    files = [file for file in files if file.split(".")[1]=="mp4"]

    result_string = ""
    for file in files:
        video = load_video_cv(image_folder_path, file)
        results = process_video_task3(video)
        
        result_string = ""
        for result in results:
            result_string = result_string + str(result) + "\n"
        result_string = result_string[:-1]

        with open(os.path.join(submission_folder_path, task_folder, file.split(".")[0] + "predicted.txt"), "w") as f:
            f.write(result_string)



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



def get_contours_for_Task1(image):
    image = get_grayscale_image(image)
    image = remove_noise_from_image_median_filter(image, 11)
    image = smoothen_image_gaussian_filter(image, 7, 1.5)
    image = edge_extraction.Canny(image, 300, 290, 5)

    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy



def filter_image(image):
    image = remove_noise_from_image_median_filter(image, 3)
    image = smoothen_image_gaussian_filter(image, 7, 1.5)
    return image


def get_arrows_from_target(image, background):
    image = filter_image(image)
    background = filter_image(background)
    arrows = extract_foreground_from_image(image, background, 35)
    return arrows

def differentiate_arrows(arrows_image):
    arrows_image = get_grayscale_image(arrows_image)
    data = list()
    for idy, line in enumerate(arrows_image):
        for idx, pixel_value in enumerate(line):
            if pixel_value != 0:
                data.append([idy, idx])
    data = np.array(data)

    from dbscan import clusterize
    result = clusterize(data)
    unique_entries = set(result)
    transform_in_color_dict = dict()

    arrows_color_list = list()
    for elem in unique_entries:
        if elem != -1:
            current_color = next(COLOR_GENERATOR)
            arrows_color_list.append(current_color)
        else:
            current_color = BLACK
        transform_in_color_dict[elem] = current_color

    unique_entries = arrows_color_list

    arrows_image = get_color_from_grayscale_image(arrows_image)
    for idx, pixel in enumerate(data):
        x_value, y_value = pixel
        arrows_image[x_value, y_value] = transform_in_color_dict[result[idx]]
    
    return arrows_image, arrows_color_list





def get_arrow_tips(arrows_image, arrows_color_list):
    arrow_tips = list()
    for color in arrows_color_list:
        arrow_points = list()
        for idy, line in enumerate(arrows_image):
            for idx, pixel_value in enumerate(line):
                pixel_value = tuple(pixel_value)
                if pixel_value == color:
                    arrow_points.append([idx, idy])
        arrow_points.sort(key=lambda elem: elem[0])
        arrow_tips.append(arrow_points[0])
    return arrow_tips




def process_image_task1(image):
    background = load_image_cv("data", "template_task1.jpg")
    image = stitch_image_inside(image, background)
    arrows_image = get_arrows_from_target(image, background)
    arrows_image = border_box_image(arrows_image, 50)
    arrows_image = remove_noise_from_image_median_filter(arrows_image, 7)
    arrows_image, arrows_color_list = differentiate_arrows(arrows_image)
    arrow_tips = get_arrow_tips(arrows_image, arrows_color_list)

    contours, hierarchy = get_contours_for_Task1(background)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    usefull_contours = list()
    for i in range(0, 16, 2):
        usefull_contours.append(contours[i])
    usefull_contours.append(contours[34])

    scores = list()
    for arrow_tip in arrow_tips:
        flag_break = False
        for idx, contour in enumerate(usefull_contours):
            if cv.pointPolygonTest(contour, arrow_tip, False) <= 0:
                scores.append(idx + 1)
                flag_break = True
                break
        
        if flag_break == False:
            scores.append(10)

    return scores

    

def process_image_task2(image):
    background = load_image_cv("data", "template_task2.jpg")
    image = stitch_image_inside(image, background)
    arrows_image = get_arrows_from_target(image, background)
    arrows_image = border_box_image(arrows_image, 50)
    arrows_image = remove_noise_from_image_median_filter(arrows_image, 7)
    arrows_image, arrows_color_list = differentiate_arrows(arrows_image)
    arrow_tips = get_arrow_tips(arrows_image, arrows_color_list)

    return ["s1" for _ in arrow_tips]
    

def process_video_task3(video):

    first_frame = None
    last_frame = None

    while True:
        ret, frame = video.read()
        
        if ret:
            last_frame = frame
        if first_frame is None:
            first_frame = frame
        if not ret:
            break
    video.release()

    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2RGB)
    last_frame = cv.cvtColor(last_frame, cv.COLOR_BGR2RGB)


    background = first_frame
    image = last_frame
    image = stitch_image_inside(image, background)
    arrows_image = get_arrows_from_target(image, background)
    arrows_image = border_box_image(arrows_image, 50)
    arrows_image = remove_noise_from_image_median_filter(arrows_image, 7)
    arrows_image, arrows_color_list = differentiate_arrows(arrows_image)
    arrow_tips = get_arrow_tips(arrows_image, arrows_color_list)

    return ["s1" for _ in arrow_tips]


if __name__ == '__main__':
    main()