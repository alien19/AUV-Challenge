# My Code 
import math
import cv2
import numpy as np


def BotPointThenUpper (Line):
     if Line[1] > Line[3] : 
        print( Line[0] , Line[1] , Line[2],Line[3] )
        return  [Line[0] , Line[1] , Line[2],Line[3]]   
     else :
        print( Line[2] , Line[3] , Line[0],Line[1] )
        return [Line[2] , Line[3] , Line[0] , Line[1]]

    



x =1

for NIMG in range(x,x+500):
    print("E:\Courses\OpenCV Learning\Test Cases\\"+ str(NIMG) + ".jpg" )
    
    frame = cv2.imread("E:\Courses\OpenCV Learning\Test Cases\\"+ str(NIMG) + ".jpg" ,-1)
    Original = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Original" , frame)
    ret = cv2.adaptiveThreshold(gray,255 ,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 9,2)
    median = cv2.medianBlur(ret,5)
    kernel= np.ones((5,5),np.float32)/25
    huh = cv2.filter2D(median,-1 , kernel)
    edges = cv2.Canny(huh, 10, 20)
 
 #Mlhaash lzma
   # erosion = cv2.dilate(edges,kernel,iterations = 1)
    
 
    # Using the Canny filter to get contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    tempContour =  []

    for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.drawContours(frame,contour, -1, (0,255, 0),1)
                cv2.imshow('Line',frame)
                # We Filter the Contour with the needed Ones
                if h*3 < w and w > 100  and h < 80 and h > 10:
                    tempContour.append(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255),2)
                    


    # Sort By Lenght
    print(len(tempContour))
    
    if len(tempContour) == 0 :
        print("Hahahahahahha")
        continue
        

    cntsSorted = sorted(tempContour, key=lambda x: cv2.arcLength(x,True))
    (x, y, w, h) = cv2.boundingRect(cntsSorted[-1])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 150, 255),2)
    
    
    #Draw The Longest Contour
 
    # then apply fitline() function
    [vx,vy,x,y] = cv2.fitLine(cntsSorted[-1],cv2.DIST_L2,0,0.01,0.01)

    # Now find two extreme points on the line to draw line
    lefty = int((-x*vy/vx) + y)
    righty = int(((gray.shape[1]-x)*vy/vx)+y)

  #  (x, y, w, h) = cv2.boundingRect(cntsSorted[-1])
    #Finally draw the line
    cv2.line(frame,(gray.shape[1]-1,righty),(0,lefty),(0,0,0),1)
    cv2.imshow("AGHAHA" , frame)
   
    cv2.imshow('Line',frame)
 
    ######################################## 
    offset = righty-5
    if offset <0 : 
        offset = 0 
    
    CheckUnderLine = gray[offset: , : ]
    cv2.imshow('Lin22211e',CheckUnderLine)
    ###############################################################################################################################
    
    edges22 = cv2.Canny(CheckUnderLine,10,20,apertureSize = 3)
   
   
    lines = cv2.HoughLinesP(image=edges22,rho=1,theta=np.pi/180, threshold=50, minLineLength=100,maxLineGap=90)
    if lines is None:
        print("Hhaahha")
        print("A = " + str(a))
    a,b,c = lines.shape
    print("A = " + str(a))
    Vert_Line = np.zeros((2,4) , dtype='i')
    TempLine = np.zeros((1,4))
    RefLine = np.zeros((1,4))
    CountVert1,CountVert2,count  = 0 ,0 ,0     # To count Vertical Lines to get Avg
    Width = 0
    for i in range(a):
        # Search For the Vertical Lines 
        # We Sum up all the Possible X,Y coords and then we take the Avg 
        #cv2.line(frame, (lines[i][0][0] , lines[i][0][1]), (lines[i][0][2] , lines[i][0][3]), (0, 255, 255), 3, cv2.LINE_AA)

        if abs(lines[i][0][1] - lines[i][0][3]) > 5 and abs(lines[i][0][0] - lines[i][0][2]) < 60:
            count = count+1
            TempLine = BotPointThenUpper(lines[i][0])
              # First Line 
            if Vert_Line[0][0] == 0 and  Vert_Line[0][1] == 0:
                #To get The bottom Point
                 Vert_Line[0] = TempLine
                 RefLine = TempLine
                 CountVert1 = CountVert1 + 1

            # Check the variance of the Horizontal axis to get the other vertical line
            elif abs(TempLine[0] - RefLine[0]) > 50:
                Vert_Line[1] =  Vert_Line[1] + TempLine
                CountVert2 = CountVert2 + 1
            
               # Width = abs(Vert_Line[0][0] - Vert_Line[1][0])
            else :    
                Vert_Line[0] =  Vert_Line[0] +TempLine
                CountVert1 = CountVert1 + 1
            
    print("_______________________________")
    print(count)
    if CountVert1 == 0 or CountVert2 == 0:
        print("Something is Wronge")
 #Create The best two Vertical lines based on the avg
    Vert_Line[0] = (Vert_Line[0]/CountVert1)            
    Vert_Line[1] = (Vert_Line[1]/CountVert2)
# 1/5 of the total Width
    Width = math.ceil(abs(Vert_Line[0][0] - Vert_Line[1][0]) *  .2 )   
    if Vert_Line[0][0] >  Vert_Line[1][0] : 
        cv2.line(frame, (Vert_Line[1][0], math.ceil(abs(Vert_Line[1][1] + Vert_Line[1][3] )/2)+righty-5), (Vert_Line[1][0] + Width, math.ceil(abs(Vert_Line[1][1] + Vert_Line[1][3] )/2)+righty-5), (255, 255, 255), 3, cv2.LINE_AA)
    else : 
        cv2.line(frame, (Vert_Line[0][0], math.ceil(abs(Vert_Line[0][1] + Vert_Line[0][3] )/2)+righty-5), (Vert_Line[0][0] + Width, math.ceil(abs(Vert_Line[0][1] + Vert_Line[0][3] )/2)+righty-5), (255, 255, 255), 3, cv2.LINE_AA)
     
    # 2 Vertical Lines    
    cv2.line(frame, (Vert_Line[0][0], Vert_Line[0][1]+righty-5), (Vert_Line[0][2], Vert_Line[0][3]+righty-5), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (Vert_Line[1][0], Vert_Line[1][1]+righty-5), (Vert_Line[1][2], Vert_Line[1][3]+righty-5), (0, 0, 255), 3, cv2.LINE_AA)
    # 1 Horizontal Line
    cv2.line(frame, (Vert_Line[1][0], righty), (Vert_Line[0][0], righty), (0, 0, 255), 3, cv2.LINE_AA)
 
    cv2.imshow("Last" , CheckUnderLine)
    #Draw Rect for The Longest Contour 

    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255),2)
    ###############################################################################################################################
 



 
    cv2.imshow("Img" , frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        cv2.imwrite("NewImg.jpg",new_img)
    if key == ord('q'):
          break
    cv2.destroyAllWindows()

cv2.destroyAllWindows()


