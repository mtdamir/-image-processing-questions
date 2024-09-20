import cv2
import numpy as np

img = cv2.imread('1.png')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for cnt in largest_contours:
    moments = cv2.moments(cnt)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        print(f"Position of star x = {center_x}, y = {center_y}")
        cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(img, "star", (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("Detect stars", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
