import cv2

image = cv2.imread('camus.jpg', cv2.IMREAD_GRAYSCALE)

# values close to 0 : darker pixels
# values close to 255 : lighter pixels

print(image.shape)
print(image)

cv2.imshow('Computer Vision', image)
cv2.waitKey(0) #wait key
cv2.destroyAllWindows() #destroy when user quit

