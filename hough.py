import cv2
import random
import numpy as np
from PIL import Image
import colorsys

def randompixel(size):
    try:
        xrand = random.randrange(0, size[0])
        yrand = random.randrange(0, size[1])
    
    except ValueError:
        return 0, 0
    
    return(xrand, yrand)


boundaries_rgb = [
    ([255, 255, 86], [255, 80, 10]),   # red 0
    # ([255, 164, 245], [150, 1, 135]),     # pink 1 
    ([255, 180, 0], [86, 255, 255]),   # yellow 2 [137,119,0],[233,214,0]
    ([177, 25, 255], [86, 255, 255])    # green 3
]
# def convertSpace():

# Yellow color range
lwrY = np.array([22, 93, 0], dtype="uint8")
uprY = np.array([45, 255, 255], dtype="uint8")

for i in range(500, 1700):
    """
    From, To
    ranges of pictures in the dataset directory
    """
    img = cv2.imread("E:/ASMarine_21/robosub_transdec_dataset/Images/"+str(i)+'.jpg')

    med = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    fltr = cv2.Canny(med, 41, 200)

    # thresh = cv2.threshold(fltr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find circles with HoughCircles
    circles = cv2.HoughCircles(fltr, cv2.HOUGH_GRADIENT, 1, minDist=img.shape[0]/10,
                    param1=200, param2=16, minRadius=13, maxRadius=30)
    

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
            buoy_pil = Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), 'RGB')
            buoy = buoy_pil.crop((x-r, y-r, x+r, y+r))
            # buoy = np.asarray(buoy)
            # avg = np.mean(b)

            pixX, pixY = randompixel(buoy.size)
            try:
                pix_b, pix_g, pix_r = buoy.getpixel((pixX, pixY))
            except IndexError:
                continue

            for ((lwr, upr), i) in zip(boundaries_rgb, range(4)):
                if (pix_b > lwr[0] and pix_b < upr[0]) and (pix_g > lwr[1] and pix_g < upr[1]) and (pix_r > lwr[2] and pix_r < upr[2]):
                    print("Done")
                    cv2.circle(img, (x, y), r, (36, 255, 12), 3)
                    # if i == 0:
                    #     cv2.putText(img, "Red buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255,0, 0))
                    # elif i == 1:
                    #     cv2.putText(img, "Orange buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (45,31, 123))
                    # elif i == 2:
                    #     cv2.putText(img, "Yellow buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (0,0, 255))
                    # elif i == 3:
                    #     cv2.putText(img, "Green buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (0,255, 0))

    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()