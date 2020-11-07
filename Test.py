import cv2
import numpy as np
from array import *



def GetBotPoint (Line):
     if Line[1] > Line[3] : 
        return  [Line[0] , Line[1]]   
     else :
        return [Line[2] , Line[3]]




def intersection( o1,  p1,  o2,  p2,
                       r):

    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross = d1.x*d2.y - d1.y*d2.x
    if abs(cross) < 1e-8 : 
        return False

    t1 = (x.x * d2.y - x.y * d2.x)/cross
    r = o1 + d1 * t1
    return True


img = cv2.imread("E:\Courses\OpenCV Learning\Test Cases\\"+ str(2) + ".jpg" ,-1)
x,y = 1,1
Point1 = np.zeros((2,2))
Point2 = np.ones((1,2))
Point1[0] = [1,2]
print(Point1[0] +Point2 )
p1 = [20,20]
o1  = [150,150]
p2 = [150,20]
o2 = [20 , 100]
r = [0,0]
#cv2.line(img , p1[0:2] , o1[0:2] ,(0,0,0),3)
#scv2.line(img , p2[0:2] , o2[0:2] ,(0,0,0),3)
cv2.imshow("Hos", img)
x  = intersection(p1,o1,p2,o2,r)
print(x)
cv2.waitKey(0)
