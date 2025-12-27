import cv2 as cv
import numpy as np

# -------------------- READ IMAGE --------------------
img = cv.imread(
    r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/211.jpg"
)

if img is None:
    raise ValueError("Image not found")

# -------------------- SHADOW HANDLING --------------------
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l = clahe.apply(l)

lab = cv.merge((l,a,b))
shadow_free = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

# -------------------- HSV GOLD MASK --------------------
hsv = cv.cvtColor(shadow_free, cv.COLOR_BGR2HSV)

lower_yellow = np.array([15, 70, 70])
upper_yellow = np.array([35, 255, 255])

mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# -------------------- MORPHOLOGICAL CLEANING --------------------
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

# -------------------- CONTOUR DETECTION --------------------
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# -------------------- BEAD FILTERING --------------------
beads = []
areas = []

for c in contours:
    area = cv.contourArea(c)
    if area > 50:
        areas.append(area)

median_area = np.median(areas)

output = img.copy()
count = 0

for c in contours:
    area = cv.contourArea(c)
    peri = cv.arcLength(c, True)

    if peri == 0:
        continue

    circularity = 4 * np.pi * area / (peri * peri)

    # -------------------- BEAD CONDITIONS --------------------
    if (
        0.4 * median_area < area < 1.8 * median_area and
        circularity > 0.6
    ):
        count += 1
        (x,y),r = cv.minEnclosingCircle(c)
        cv.circle(output, (int(x),int(y)), int(r), (0,255,0), 2)
        cv.putText(output, str(count), (int(x)-5, int(y)-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# -------------------- RESULTS --------------------
print("Final Bead Count:", count)

cv.imshow("Original", img)
cv.imshow("Shadow Free", shadow_free)
cv.imshow("Gold Mask", mask)
cv.imshow("Final Bead Detection", output)

cv.waitKey(0)
cv.destroyAllWindows()
