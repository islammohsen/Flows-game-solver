import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# read image
img = cv.imread('example.jpg')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# edge detection
edges = cv.Canny(gray, 50, 150)
kernel = np.ones((5, 5), np.uint8)
edges = cv.dilate(edges, kernel, iterations=1)

# contours detection
contours, hierarchy = cv.findContours(
    edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# drawing bounding boxes
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    img = cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# ploting
plt.imshow(img)
plt.show()
