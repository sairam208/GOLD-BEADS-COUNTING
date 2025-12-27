import cv2 as cv
import numpy as np

# -------------------- READ IMAGE --------------------
img_org = cv.imread(
    r"G:/PROJECTS/IMAGE PROCESSING/GOLD PROJECT/kannan p1 2/kannan p1 2/gold/7/211.jpg"
)

if img_org is None:
    raise ValueError("Image not found")

blur = cv.GaussianBlur(img_org, (5, 5), 0)
lab = cv.cvtColor(blur, cv.COLOR_BGR2LAB)

# -------------------- DIFFERENT LAB GOLD RANGES TO TEST --------------------
lab_ranges = [
    ("Range-1", np.array([20, 118, 130]), np.array([255, 150, 175])),
    ("Range-2", np.array([20, 120, 135]), np.array([255, 155, 180])),
    ("Range-3", np.array([30, 120, 140]), np.array([255, 160, 185])),
    ("Range-4", np.array([25, 115, 140]), np.array([255, 150, 190])),
    ("Range-5", np.array([15, 110, 125]), np.array([255, 160, 195])),
]

# -------------------- LOOP THROUGH RANGES --------------------
for name, lower_gold, upper_gold in lab_ranges:

    # ---- LAB MASK ----
    lab_mask = cv.inRange(lab, lower_gold, upper_gold)

    # ---- GRAPHCUT ----
    gc_mask = np.zeros(img_org.shape[:2], np.uint8)
    gc_mask[:] = cv.GC_PR_BGD
    gc_mask[lab_mask > 0] = cv.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv.grabCut(
        img_org,
        gc_mask,
        None,
        bgdModel,
        fgdModel,
        5,
        cv.GC_INIT_WITH_MASK
    )

    final_mask = np.where(
        (gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    # ---- MASK CLEANING ----
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel)

    # ---- APPLY MASK TO ORIGINAL IMAGE ----
    masked_img = cv.bitwise_and(img_org, img_org, mask=final_mask)

    # ---- HSV REFINEMENT ----
    hsv = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    hsv_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
    hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_CLOSE, kernel)

    # ---- EDGE DETECTION ----
    edges = cv.Canny(hsv_mask, 50, 150)

    # ---- DISPLAY ----
    cv.imshow("Original", img_org)
    cv.imshow(f"{name} - LAB Mask", lab_mask)
    cv.imshow(f"{name} - GraphCut Mask", final_mask)
    cv.imshow(f"{name} - Masked Original", masked_img)
    cv.imshow(f"{name} - Edges", edges)

    print(f"Showing results for {name}")
    cv.waitKey(0)
    cv.destroyAllWindows()
