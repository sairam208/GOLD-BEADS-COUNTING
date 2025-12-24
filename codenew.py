import cv2 as cv
import numpy as np

# -------------------- READ IMAGE --------------------
img = cv.imread(
    r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/240.jpg"
)

# -------------------- HSV MASK --------------------
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_yellow = np.array([15, 80, 80])
upper_yellow = np.array([35, 255, 255])

mask = cv.inRange(hsv, lower_yellow, upper_yellow)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

# -------------------- EDGE DETECTION --------------------
edges = cv.Canny(mask, 50, 150)

# -------------------- DISTANCE TRANSFORM (NECK FINDING) --------------------
dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
dist_norm = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

_, necks = cv.threshold(dist_norm, 40, 255, cv.THRESH_BINARY_INV)
necks = cv.bitwise_and(necks, mask)

# -------------------- BREAK MERGED CONTOURS --------------------
broken_mask = mask.copy()
broken_mask[necks == 255] = 0
broken_mask = cv.morphologyEx(broken_mask, cv.MORPH_OPEN, kernel)

# -------------------- FIND CONTOURS --------------------
contours, _ = cv.findContours(
    broken_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

# ===================================================
# ADAPTIVE BEAD AREA SELECTION (ALTERED PART)
# ===================================================

areas = []
valid_contours = []

# Collect contour areas
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 10:   # remove tiny noise
        areas.append(area)
        valid_contours.append(cnt)

areas = np.array(areas)

beads = []

if len(areas) > 0:
    # Most common bead area (median)
    common_area = np.median(areas)

    # Adaptive range (±40%)
    lower_area = 0.4 * common_area
    upper_area = 2 * common_area

    print("Common bead area:", common_area)
    print("Adaptive range:", lower_area, "-", upper_area)

    # Select beads
    for cnt in valid_contours:
        area = cv.contourArea(cnt)
        if lower_area <= area <= upper_area:
            beads.append(cnt)
print("Number of beads:", len(beads))

# -------------------- DRAW CENTER, CIRCLE & NUMBER --------------------
output = img.copy()
font = cv.FONT_HERSHEY_SIMPLEX

for i, cnt in enumerate(beads, start=1):

    M = cv.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    (x, y), radius = cv.minEnclosingCircle(cnt)

    cv.circle(output, (int(x), int(y)), int(radius), (255, 0, 0), 2)
    cv.circle(output, (cx, cy), 3, (0, 0, 255), -1)

    cv.putText(
        output,
        str(i),
        (cx - 10, cy - 10),
        font,
        0.5,
        (0, 0, 255),
        1,
        cv.LINE_AA
    )

# -------------------- DISPLAY --------------------
cv.imshow("Original Image", img)
cv.imshow("HSV Mask", mask)
cv.imshow("Neck Regions", necks)
cv.imshow("Broken Mask", broken_mask)
cv.imshow("Final Beads (Adaptive Area)", output)

cv.waitKey(0)
cv.destroyAllWindows()
