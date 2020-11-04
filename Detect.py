# My Code 


import cv2
import numpy as np


x =1

for i in range(x,x+500):
    #print("E:\Courses\OpenCV Learning\Test Cases\\"+ str(i) + ".jpg" )
    frame = cv2.imread("E:\Courses\OpenCV Learning\Test Cases\\"+ str(i) + ".jpg" ,-1)
    Original = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,5)
    
    ret = cv2.adaptiveThreshold(median,255 ,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 9,2)
    

    kernel = np.ones((2,2),np.uint8)

    edges = cv2.Canny(ret, 10, 20)
 
    erosion = cv2.dilate(edges,kernel,iterations = 1)
    
 
    # Using the Canny filter to get contours
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    """
        for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if cv2.contourArea(contour) <200:
                    continue
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)
                
    """ 
  
    # Sort By Lenght
    cntsSorted = sorted(contours, key=lambda x: cv2.arcLength(x,True))
    
    print("Number of contours = " + str(len(contours)),)
    
    #Draw The Longest Contour
    cv2.drawContours(frame, cntsSorted[-1], -1, (0, 0, 0), 3)
 
 
    # then apply fitline() function
    [vx,vy,x,y] = cv2.fitLine(cntsSorted[-1],cv2.DIST_L2,0,0.01,0.01)

# Now find two extreme points on the line to draw line
    lefty = int((-x*vy/vx) + y)
    righty = int(((gray.shape[1]-x)*vy/vx)+y)
# 
    (x, y, w, h) = cv2.boundingRect(cntsSorted[-1])
#Finally draw the line
    cv2.line(frame,(gray.shape[1]-1,righty),(0,lefty),255,2)
   # crop = frame[y:y+h+20,x-30:x+w]
    cv2.imshow('Line',frame)
    print((gray.shape[1]-1,righty))
    print((0,lefty))
    ######################################## 
    vertical = np.copy(Original)
    
    CheckUnderLine = gray[righty-5: , : ]
    
    
    edges22 = cv2.Canny(CheckUnderLine,10,20,apertureSize = 3)
   # cv2.imshow("Yalaaaa" , edges22)
    minLineLength=100
    lines = cv2.HoughLinesP(image=edges22,rho=1,theta=np.pi/180, threshold=50,lines=np.array([]), minLineLength=minLineLength,maxLineGap=90)
    a,b,c = lines.shape
    
    
    for i in range(a):
        #Search For the Vertical Lines 
        if abs(lines[i][0][1] - lines[i][0][3]) > 5 and abs(lines[i][0][0] - lines[i][0][2]) < 20:
             cv2.line(CheckUnderLine, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 3, cv2.LINE_8)
             
             print("Finlayyyyyy")
             
        #cv2.line(CheckUnderLine, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Last" , CheckUnderLine)
    #Draw Rect for The Longest Contour 
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255),2)
    
  #  print(cv2.arcLength(cntsSorted[-2],True))
    cv2.drawContours(frame, cntsSorted[-9:-2], -1, (0, 255, 0), 3)
  #Crop
   
   # cv2.imshow("Crr" , new_img)
    cv2.imshow("Img" , frame)
   # cv2.imshow("Orgggg" , Original)
    #cv2.imshow("Img2" , frame2)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        cv2.imwrite("NewImg.jpg",new_img)
    if key == ord('q'):
          break
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
