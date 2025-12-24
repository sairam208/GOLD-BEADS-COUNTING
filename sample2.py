# Sharp close + split beads and pendant into separate contours

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread(
    r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/223.jpg"
)
if img is None:
    raise FileNotFoundError(
        "Failed to load image. Check the image path in sample2.py"
    )

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
threshold, thresh = cv.threshold(
    gray, 70, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    if i == 0:
        continue

    epsilon = 8.81 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    cv.drawContours(img, contour, 0, (0, 0, 0), 4)  
cv.imshow('thresh', thresh)
cv.imshow("Contour Approximation", img)
cv.waitKey(0)
cv.destroyAllWindows()