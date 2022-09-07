import numpy as np
from sklearn.cluster import KMeans

def segment_image_based_on_color_KMeans(image, number_of_colors):
    assert len(image.shape) == 3, f"Image must be coloured. Image shape is not a 2 by 2 RGB. Given is {image.shape}"

    height, width, depth = image.shape

    image = image.reshape(height * width, depth)

    kmeans = KMeans(number_of_colors)
    labels = kmeans.fit_predict(image)
    means = kmeans.cluster_centers_

    dict_means = dict()
    for id, mean in enumerate(means):
        dict_means[id] = list(mean)

    segmented_image = np.array([dict_means[elem] for elem in labels], dtype=np.uint8)
    segmented_image = segmented_image.reshape(height, width, depth)

    return segmented_image