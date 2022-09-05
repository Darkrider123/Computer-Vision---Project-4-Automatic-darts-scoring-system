import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def show_image(image, grayscale=False, maximize=False):
    if maximize is True:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

    if grayscale is True:
        plt.imshow(np.uint8(image), cmap='gray')
    else:
        plt.imshow(np.uint8(image))
    plt.show()
    

def show_image_cv(image, image_name="image"):
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.imshow(image_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def select_ROI(image):
    roi = cv.selectROI(image)
    cv.destroyAllWindows()
    return roi


def draw_circle(image, circle, color, thickness=3):
    assert len(circle) == 3
    x, y, radius = circle
    x = int(x)
    y = int(y)
    radius = int(radius)
    cv.circle(image, (x, y), radius, color, thickness)