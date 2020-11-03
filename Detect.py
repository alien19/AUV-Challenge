# My Code 


import cv2
import numpy as np


x =1

for i in range(x,x+500):
    #print("E:\Courses\OpenCV Learning\Test Cases\\"+ str(i) + ".jpg" )
    frame = cv2.imread("E:\Courses\OpenCV Learning\Test Cases\\"+ str(i) + ".jpg" ,-1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,5)
    cv2.imshow("Blur" , median)
    ret = cv2.adaptiveThreshold(median,255 ,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 9,2)
    cv2.imshow("Thersh" , ret)

    kernel = np.ones((2,2),np.uint8)

    edges = cv2.Canny(ret, 10, 20)
    cv2.imshow("Canny1" , edges)
    erosion = cv2.dilate(edges,kernel,iterations = 1)
    
    cv2.imshow("Open" , erosion)
    # Using the Canny filter to get contours
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours2, hierarchy2 = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    """
        for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if cv2.contourArea(contour) <200:
                    continue
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)
                
    """ 
  
    frame2 = frame
    cntsSorted = sorted(contours, key=lambda x: cv2.arcLength(x,True))
    print("Number of contours = " + str(len(contours)), len(contours[-1]))
    
    cv2.drawContours(frame, cntsSorted[-1], -1, (0, 0, 0), 3)
    
    
    (x, y, w, h) = cv2.boundingRect(cntsSorted[-1])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255),2)
    
    cv2.drawContours(frame, cntsSorted[-9:-3], -1, (0, 255, 0), 3)
    #cv2.drawContours(frame2, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Img" , frame)
    cv2.imshow("Img2" , frame2)
    edges = cv2.Canny(ret, 10, 20)
    cv2.imshow("Canny1" , edges)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
          break
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
