import cv2
import numpy as np
for i in range(1000, 1653):
    img = cv2.imread("E:/ASMarine_21/robosub_transdec_dataset/Images/"+str(i)+'.jpg')
    # Load img, grayscale, Otsu's threshold
    med = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    fltr = cv2.Canny(med, 41, 200)

    # thresh = cv2.threshold(fltr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find circles with HoughCircles
    circles = cv2.HoughCircles(fltr, cv2.HOUGH_GRADIENT, 1, minDist=img.shape[0]/10,
                    param1=200, param2=16, minRadius=2, maxRadius=30)
    # Draw circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if(r > 15 and r <= 52):
                cv2.circle(img, (x, y), r, (36, 255, 12), 3)

    # cv2.imshow('thresh', thresh)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()