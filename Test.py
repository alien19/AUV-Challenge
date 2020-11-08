import cv2
import numpy as np
from array import *
import math


def GetBotPoint (Line):
     if Line[1] > Line[3] : 
        return  [Line[0] , Line[1]]   
     else :
        return [Line[2] , Line[3]]


# Intersection between 2 Lines <3 
def TwoLineIntersection( o1,  p1,  o2,  p2):

    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross =  d1[0]*d2[1] - d1[1]*d2[0]
    print(cross)
    if abs(cross) < 1e-8 : 
        return False ,[0,0]

    t1 = (x[0]* d2[1] - x[1] * d2[0])/cross
    r = o1 + d1 * t1
    return True , r.astype(int)


img = cv2.imread("E:\Courses\OpenCV Learning\Test Cases\\"+ str(2) + ".jpg" ,-1)
x,y = 1,1
Point1 = np.zeros((2,2))
Point2 = np.ones((1,2))
Point1[0] = [1,2]
print(Point1[0] +Point2 )
p1 = np.array([50,10])
o1  = np.array([50,40])
p2 = np.array([60,80])
o2 = np.array([60 , 40])
r = np.array([])
cv2.line(img , (p1[0] , p1[1] ) , (o1[0] , o1[1] ) ,(0,0,0),3)
cv2.line(img , (p2[0] , p2[1] ) ,(o2[0] , o2[1] ) ,(0,0,0),3)

Flag,r  =  intersection(o1,p1,p2,o2)

cv2.imshow("Hos", img)
print(Flag)
print(r)
cv2.waitKey(0)
