import cv2
import numpy as np

# 1 - remove the vertical line on the left

img_org = cv2.imread('images/multi_digits.png', 0)

# step1
edges = cv2.adaptiveThreshold(img_org, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# step2
kernel = np.ones((1, 2), dtype="uint8")
dilated = cv2.dilate(edges, kernel)
cv2.imshow("dilated", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

ctrs, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

# inverse color
# img = cv2.bitwise_not(img_org)

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    print(x, y, x + w, y + h)

    # Getting ROI
    roi = img_org[y:y + h, x:x + w]
    cv2.imwrite(f'{i}.png', roi)
