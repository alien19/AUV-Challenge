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
    ([175, 50, 20], [180, 255, 255]),   # red 0
    ([255, 164, 245], [150, 1, 135]),     # pink 1 
    ([0, 180, 255], [170, 255, 255]),   # yellow 2 [137,119,0],[233,214,0]
    ([36, 25, 25], [86, 255, 255])    # green 3
]

# def loadFrames(vid):
cap = cv2.VideoCapture("video_without.mp4")
while cap.isOpened():
    _, frame = cap.read()
    med = cv2.medianBlur(frame, 5)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    fltr = cv2.Canny(med, 41, 200)

    circles = cv2.HoughCircles(fltr, cv2.HOUGH_GRADIENT, 1, minDist=frame.shape[0]/10,
                param1=200, param2=16, minRadius=12, maxRadius=30)
                
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        buoy_hsv = []
        for (x, y, r) in circles:
            # if(r > 15 and r <= 52):    # relatively useless
            buoy_pil = Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)), 'HSV')
            buoy = buoy_pil.crop((x-r, y-r, x+r, y+r))
            

            pixX, pixY = randompixel(buoy.size)
            try:
                pix_h, pix_s, pix_v = buoy.getpixel((pixX, pixY))
            except IndexError:
                continue

            for ((lwr, upr), i) in zip(boundaries_rgb, range(4)):
                if (pix_h > lwr[0] and pix_h < upr[0]) and (pix_s > lwr[1] and pix_s < upr[1]) and (pix_v > lwr[2] and pix_v < upr[2]):
                    print("Done")
                    cv2.circle(frame, (x, y), r, (36, 255, 12), 3)
                    if i == 0:
                        cv2.putText(frame, "Red buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255,0, 0))
                    elif i == 1:
                        cv2.putText(frame, "Pink buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7, (255,255, 255))
                    elif i == 2:
                        cv2.putText(frame, "Yellow buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7, (255,255, 255))
                    elif i == 3:
                        cv2.putText(frame, "Green buoy", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7, (0,255, 0))


        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()