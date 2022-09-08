import numpy as np
import cv2

original_image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

# firstly we have to convert the image into grayscale
# opencv handles BGR instead of RGB
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Laplacian Kernel for edge detection
# So instead of this
# kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
# result_image = cv2.filter2D(gray_image, -1, kernel)

# We can use the laplacian kernel to detect edges
result_image = cv2.Laplacian(gray_image, -1)

cv2.imshow('Original Image', original_image)
cv2.imshow('Gray image', gray_image)
cv2.imshow('Result image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
