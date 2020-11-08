import cv2
import random
import numpy as np
from PIL import Image


def randompixel(size):
    try:
        xrand = random.randrange(0, size[0])
        yrand = random.randrange(0, size[1])
    
    except ValueError:
        return 0, 0
    
    return(xrand, yrand)


boundaries_rgb = [
    ([175, 50, 20], [180, 255, 255]),   # red
    # ([35, 140, 60], [255, 255, 180]),   # blue
    ([17, 15, 100], [50, 56, 200]),     # pink
    ([0, 180, 255], [170, 255, 255]),   # yellow
    ([36, 25, 25], [86, 255, 255])    # green
]

# Yellow color range
lwrY = np.array([22, 93, 0], dtype="uint8")
uprY = np.array([45, 255, 255], dtype="uint8")

for i in range(1000, 1700):
    """
    ranges of pictures in the dataset directory
    """
    img = cv2.imread("E:/ASMarine_21/robosub_transdec_dataset/Images/"+str(i)+'.jpg')

    med = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    fltr = cv2.Canny(med, 41, 200)

    # thresh = cv2.threshold(fltr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find circles with HoughCircles
    circles = cv2.HoughCircles(fltr, cv2.HOUGH_GRADIENT, 1, minDist=img.shape[0]/10,
                    param1=200, param2=16, minRadius=12, maxRadius=30)
    

    # buoy_hsv = cv2.cvtColor(buoy, cv2.COLOR_BGR2HSV)
    # ## Detecting the yellow buoy 
    # # Yellow color range
    # lwrY = np.array([22, 93, 0], dtype="uint8")
    # uprY = np.array([45, 255, 255], dtype="uint8")
    
    # Draw circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        buoy_hsv = []
        for (x, y, r) in circles:
            # if(r > 15 and r <= 52):    # relatively useless
            buoy_pil = Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)), 'HSV')
            buoy = buoy_pil.crop((x-r, y-r, x+r, y+r))
            # buoy.show()
            # print(buoy.size, buoy_pil.size)
            pixX, pixY = randompixel(buoy.size)
            try:
                pix_h, pix_s, pix_v = buoy.getpixel((pixX, pixY))
            except IndexError:
                continue

            for (lwr, upr) in boundaries_rgb:
                if (pix_h > lwr[0] and pix_h < upr[0]) and (pix_s > lwr[1] and pix_s < upr[1]) and (pix_v > lwr[2] and pix_v < upr[2]):
                    print("Done")
                    cv2.circle(img, (x, y), r, (36, 255, 12), 3)

    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()