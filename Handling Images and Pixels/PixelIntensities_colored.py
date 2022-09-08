import cv2
import numpy as np

image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)
print(image.shape)
print(image)
print(np.amax(image))
print(np.amin(image))
cv2.imshow('COMPUTER VISION',image)
cv2.waitKey(0)
cv2.destroyAllWindows()