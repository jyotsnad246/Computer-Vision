import numpy as np
import cv2

original_image = cv2.imread('unsharp_bird.jpg', cv2.IMREAD_COLOR)

#sharpen kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpen_image = cv2.filter2D(original_image, -1, kernel)

cv2.imshow('Original Image', original_image)
cv2.imshow('Sharpen Image', sharpen_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# convolution is the process of adding each element of the image to its local neighbors - weighted by the kernel
