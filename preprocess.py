import cv2
from utils_show import imshow_r, to_bgr
import numpy as np
from PIL import Image

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def preprocess_legend_coordinates(points):
    """ 
    Given coordinates of a legend, 
    ensure that we have top-left and bottom-right coordinates of the legend
    """
    # Get the outrmost points if coordinates of a polygon are given
    if len(points) > 2:
        points = bounding_box(points)

    # Sort points by x-axis
    points = sorted(points, key=lambda x : x[0])

    # Swap y coors, when we have (bottom-left, top-right) 
    # instead of (top-left, bottom-right) as corners of the legend,
    # [0, 1        [1, 0]
    #  1, 0]   --> [0, 1]
    if points[0][1] > points[1][1]:
        points[0][1], points[1][1] = points[1][1], points[0][1]

    # points = [(int(point[0]), int(point[1]))for point in points]

    return points


def preprocess_legend(legend):
    if not isinstance(legend, np.ndarray):
        legend = cv2.cvtColor(np.array(legend), cv2.COLOR_RGB2BGR)
        legend_original = legend.copy()

    assert (legend.shape[-1] == 3), "Legend is expected in RGB format loaded from PIL"
    legend_gray = cv2.cvtColor(legend, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    # legend_gray = cv2.GaussianBlur(legend_gray, (3,3), 0)

    # Morph open (erosion followed by dilation)
    legend_gray = cv2.morphologyEx(legend_gray, cv2.MORPH_OPEN, (2, 2))
    # imshow_r('legend_gray', list([legend, to_bgr(legend_gray)]), True)

    # Threshold the image to make it binary using OSTU (didn't really work well. We'll use a static threshold)
    # (thresh, legend_gray) = cv2.threshold(legend_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    thresh = 50
    legend_gray = cv2.threshold(legend_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # imshow_r('threshold', list([to_bgr(legend_original), to_bgr(legend_gray)]), True)
    legend_processed = cv2.bitwise_and(legend_original, legend_original, mask = legend_gray)
    imshow_r('legend_processed', legend_processed, True)
    
    legend_processed = Image.fromarray(legend_processed)
    legend_processed.show()
    return legend_gray