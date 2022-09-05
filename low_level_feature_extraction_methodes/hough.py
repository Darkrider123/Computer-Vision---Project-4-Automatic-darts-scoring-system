import cv2 as cv

def get_circles_from_image(image, start_radius, end_radius, increment):
    assert len(image.shape) == 2 , f"Image must be grayscale. Image shape is not a 2 by 2, it is {image.shape}"

    circles = list()
    radiuses = [radius for radius in range(start_radius, end_radius, increment)]
    for minRadius, maxRadius in zip(radiuses[:-1], radiuses[1:]):
        minRadius = minRadius + 1

        circles_with_given_radius = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1,5, param1=25,param2=75, minRadius=minRadius, maxRadius=maxRadius)

        if circles_with_given_radius is not None:
            for circle in circles_with_given_radius:
                circles.append(circle[0])
    
    return circles