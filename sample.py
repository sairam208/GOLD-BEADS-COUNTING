import cv2
import numpy as np
img = cv2.imread(
    r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/240.jpg"
)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):  
    if i == 0:
        continue

    epsilon = 8.81 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours (img, contour, 0, (0,0,0), 4)
    cv2.imshow("Contour Approximation", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()