import cv2
import numpy as np


def draw_the_lines(image, lines):
    # create a distinct image for lines [0,255] -  all 0 values mean black image
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are (x,y) coordinates for the starting and end points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line: #BGR - so blue line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # merge image with lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)

    return image_with_lines


def region_of_interest(image, region_points):
    # we are going to replace pixels with 0 (black) - regions we are not interested
    mask = np.zeros_like(image) #mask will have the exact same dimensions as the image
    # lower triangle - 255 pixels
    cv2.fillPoly(mask, region_points, 255)
    # keep the regions of the original image where the mask has white color
    mask_image = cv2.bitwise_and(image, mask)

    return mask_image

def get_detected_lanes(image):

    (height, width) = (image.shape[0], image.shape[1])

    # we have to turn the image into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)

    # we are interested int the lower region of the image (because we only need the driving lanes)
    region_of_interest_vertices = [
        # define 3 points of interest
        (0, height),
        (width/2, height*0.65),
        (width, height)
    ]

    # we can get rid of the un relevant part of the image we just keep the lower triangle region
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices],np.int32))

    # using line detection algorithm - hough transformation ( we are dealing with radians instead of degrees here)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=150)
    # draw on the image
    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines

# video = several frames (image shown right after each other)
video = cv2.VideoCapture('lane_detection_video.mp4')
while video.isOpened():
    is_grabbed, frame = video.read()
    # for the end of the video
    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)
    cv2. imshow('Lane Detection Video', frame)
    cv2.waitKey(40)
video.release()
cv2.destroyAllWindows()

# Canny's Algorithm
# -> it can detect edges like Laplacian
# -> it is an optimal detector, has low error rate
# -> can detect existing edges only
# -> it has good localization
# -> every edge is detected only once


# https://www.researchgate.net/post/What-is-a-useful-definition-of-an-edge-in-image-processing

# see why we use bitwise_and from the course


# Hough Algorithm
#

# image to grayscale
# edge detection kernel
# lower triangle region
# High transformation
# draw lines on image



