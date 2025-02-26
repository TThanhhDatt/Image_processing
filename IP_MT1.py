# Task 1
import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture("task1.mp4")

output = cv2.VideoWriter('task1_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20.0, (int(video.get(3)), int(video.get(4))))

while video.isOpened():
  ret, frame = video.read()
  if not ret:
    break

  frame = cv2.putText(frame, '52100879', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
  #Convert video to HSV space
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Set up mask for traffic sign with red and blue color
  lower_red = np.array([0, 50, 50])
  upper_red = np.array([10, 255, 255])
  mask1 = cv2.inRange(hsv, lower_red, upper_red)
  lower_red = np.array([170, 50, 50])
  upper_red = np.array([180, 255, 255])
  mask2 = cv2.inRange(hsv, lower_red, upper_red)
  mask_red = mask1 + mask2
  lower_blue = np.array([100, 150, 100])
  upper_blue = np.array([140, 255, 255])
  mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
  mask = cv2.bitwise_or(mask_red, mask_blue)

  # Using GaussianBlur to smooth image and noise reduction
  blur = cv2.GaussianBlur(mask, (5, 5), 0)

  # Detect edges and find contours
  edges = cv2.Canny(blur, 50, 150)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
    if len(approx) > 4:
      (x,y), radius = cv2.minEnclosingCircle(cnt)
      center = (int(x), int(y))
      radius = int(radius)
      contour_area = cv2.contourArea(cnt)
      bounding_box_area = (2 * radius) ** 2
      area_ratio = contour_area / bounding_box_area
      if 0.6 < area_ratio < 2.0 and radius > 25 and radius < 60:
        cv2.rectangle(frame, (center[0] - radius, center[1] - radius),(center[0]
                      + radius, center[1] + radius), (0, 255, 0), 2)


  output.write(frame)
video.release()
output.release()
cv2.destroyAllWindows()

# Task 2
img = cv2.imread("input.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blur = cv2.GaussianBlur(gray, (5, 5), 0)

img_scales = cv2.convertScaleAbs(img_blur, alpha=1.5, beta=0)
ret, thresh = cv2.threshold(img_scales, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None] * len(contours)
boundRect =[]
img_copy = img.copy()
for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1:
        contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
        boundRect.append(cv2.boundingRect(contours_poly[i]))
for i in range(len(boundRect)):
    cv2.rectangle(img_copy, (int(boundRect[i][0]), int(boundRect[i][1])), \
              (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), (0,255,0), 2)

cv2.imwrite("task2_output.png", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()