import cv2 as cv
import numpy as np

IMG_PATH = r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/240.jpg"

img = cv.imread(IMG_PATH)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

if img is None:
    raise FileNotFoundError(f"Failed to load image: {IMG_PATH}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
lower_yellow = np.array([15, 80, 80])
upper_yellow = np.array([35, 255, 255])

mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Otsu threshold (from sample2.py)
blur = cv.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)

# Find edges from the thresholded image
# Canny works on 8-bit single-channel images
edges = cv.Canny(thresh, 50, 150)

# Create a colored overlay of edges (red) on the original image
colored_edges = np.zeros_like(img)
colored_edges[edges != 0] = (0, 0, 255)

overlay = cv.addWeighted(img, 0.8, colored_edges, 0.6, 0)

# Optional: find contours on thresh for further processing
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours (for visualization) on a copy
cont_vis = img.copy()
cv.drawContours(cont_vis, contours, -1, (0, 255, 0), 1)

# Display results
cv.imshow("Original", img)
cv.imshow("Otsu Thresh", thresh)
cv.imshow("Edges from Thresh", edges)
cv.imshow("Edges Overlay", overlay)
cv.imshow("Contours on Orig", cont_vis)

print(f"Found {len(contours)} contours from threshold.")

cv.waitKey(0)
cv.destroyAllWindows()


