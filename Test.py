import cv2
import numpy as np
from array import *



def GetBotPoint (Line):
     if Line[1] > Line[3] : 
        return  [Line[0] , Line[1]]   
     else :
        return [Line[2] , Line[3]]



x,y = 1,1
Point1 = np.zeros((2,2))
Point2 = np.ones((1,2))
Point1[0] = [1,2]
print(Point1[0] +Point2 )


