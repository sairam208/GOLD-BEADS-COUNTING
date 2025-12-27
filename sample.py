import cv2
import numpy as np
import os
import sys

# 1. Preprocessing
# default path (can be overridden by passing a path as first arg)
default_name = r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/211.jpg"
input_path = sys.argv[1] if len(sys.argv) > 1 else default_name
if not os.path.isabs(input_path):
    input_path = os.path.join(os.path.dirname(__file__), input_path)

if not os.path.exists(input_path):
    print(f"Error: image not found at {input_path}")
    sys.exit(1)

image = cv2.imread(input_path)
if image is None:
    print(f"Error: cv2.imread returned None for {input_path} — file may be unreadable or corrupted")
    sys.exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

# 2. Find Contours
# findContours can pick up incomplete arcs if they are sufficiently long
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # 3. Filter by size to ignore noise
    if len(cnt) < 5:  # fitEllipse requires at least 5 points
        continue
        
    # 4. Fit the Ellipse
    # fitEllipse uses Least Squares to "complete" the shape even if it's an arc
    ellipse = cv2.fitEllipse(cnt)
    
    # 5. Check "Circular" or "Oval"
    (x, y), (MA, ma), angle = ellipse
    # skip degenerate ellipses (invalid sizes)
    if not (np.isfinite(MA) and np.isfinite(ma) and MA > 0 and ma > 0):
        continue

    ratio = min(MA, ma) / max(MA, ma)

    # ratio close to 1.0 = Circle; lower = Oval
    color = (0, 255, 0) if ratio > 0.9 else (255, 0, 0)

    cv2.ellipse(image, ellipse, color, 2)

cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()