import os
from os import path, walk

import cv2
import numpy as np
import imutils
from imutils import contours

folders = os.listdir()
cv_img = []
images = {}
for root, dirs, files in walk("E:\Courses\OpenCV Learning\Test Cases\"):
    for file, i in zip(files, range(100)):
        if file.endswith(".jpg"):
            images[file] = cv2.imread(os.path.join(root, file))
            # img = cv2.imread(os.path.join(root, file))      # 0 for gray scale &
            # cv2.imshow("image"+str(i), img)
            # cv_img.append(img)
import glob
import re
files = glob.glob("E:\Courses\OpenCV Learning\Test Cases\*.jpg")
files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[1]))
cv_imgs = []
for path, i in zip(files, range(100)):
    img = cv2.imread(path)
    cv_imgs.append(img)
    # cv2.imshow("image"+str(i), img)
    # cv2.waitKey(0)

for img in cv_imgs:
    fltr = cv2.Canny(img, 41, 100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    # ret, thresh = cv2.threshold(th3, 100, 255, cv2.THRESH_BINARY)
    med = cv2.medianBlur(th3, 5)
    contours, hierarchy = cv2.findContours(med, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
    
    cv2.imshow("Original", img)
    cv2.imshow("Contours", med)
    # cv2.imshow("Adaptive", th3)
    cv2.waitKey(0)

cv2.destroyAllWindows()
