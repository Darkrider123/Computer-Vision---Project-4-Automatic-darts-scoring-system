import cv2 as cv
import numpy as np

def get_circles_from_image(image, start_radius, end_radius, increment, canny_threshold, precision, accumulator_size_meaning_accuracy = 1, minimum_distance_between_circle_centers=5, method=cv.HOUGH_GRADIENT):
    assert len(image.shape) == 2 , f"Image must be grayscale. Image shape is not a 2 by 2. Given is {image.shape}"
    assert method == (cv.HOUGH_GRADIENT or cv.HOUGH_GRADIENT_ALT), f"Method must be HOUGH_GRADIENT or HOUGH_GRADIENT_ALT. Given is {method}"

    circles = list()
    radiuses = [radius for radius in range(start_radius, end_radius, increment)]
    for minRadius, maxRadius in zip(radiuses[:-1], radiuses[1:]):
        minRadius = minRadius + 1

        circles_with_given_radius = cv.HoughCircles(image=image, method=method, dp=accumulator_size_meaning_accuracy, minDist=minimum_distance_between_circle_centers, param1=canny_threshold,param2=precision, minRadius=minRadius, maxRadius=maxRadius)

        if circles_with_given_radius is not None:
            for circle in circles_with_given_radius:
                circles.append(circle[0])
    
    if len(circles) == 0:
        return None
    return circles

def get_lines_from_image(image, threshold=50, minLineLength=0, maxLineGap=0, rho=1, theta=np.pi/180, lines=None):
    assert len(image.shape) == 2 , f"Image must be grayscale. Image shape is not a 2 by 2, it is {image.shape}"

    h_lines = cv.HoughLinesP(image, rho=rho,theta=theta, threshold=threshold, lines=lines, minLineLength=minLineLength,maxLineGap=maxLineGap)

    lines = None
    if h_lines is not None:
        lines = [line[0] for line in h_lines]
    
    return lines