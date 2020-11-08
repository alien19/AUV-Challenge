# My Code 
import math
import cv2
import numpy as np
import time



# Function Takes 2 points and rearrange them 
def BotPointThenUpper (Line):
    #Compare Y values to get the Bot point first
     if Line[1] > Line[3] : 
       # print( Line[0] , Line[1] , Line[2],Line[3] )
        return  [Line[0] , Line[1] , Line[2],Line[3]]   
     else :
       # print( Line[2] , Line[3] , Line[0],Line[1] )
        return [Line[2] , Line[3] , Line[0] , Line[1]]

# Function Takes 2Lines A(o1,p1) B(o2,p2) and return True if Intersected and the X,Y Coodrs of the Intersection
def TwoLineIntersection( o1,  p1,  o2,  p2):

    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross =  d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-8 : 
        return False ,[0,0]

    t1 = (x[0]* d2[1] - x[1] * d2[0])/cross
    r = o1 + d1 * t1
    return True , r.astype(int)

    
def DetectGate(Original):
    frame = np.copy(Original)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# __________Filters_____________
   # cv2.imshow("Original" , frame)
    ret = cv2.adaptiveThreshold(gray,255 ,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 9,2)
    median = cv2.medianBlur(ret,5)
    kernel= np.ones((5,5),np.float32)/25
    Filter= cv2.filter2D(median,-1 , kernel)
    edges = cv2.Canny(Filter, 10, 20)
 
# _______ Contours _____________
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tempContour =  []
    #Loop throught the Contours to Find the needed ones
    for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
               # cv2.drawContours(frame,contour, -1, (0,255, 0),1)
               # cv2.imshow('Line',frame)
                # We Filter the Contour with the needed Ones
                if h*3 < w and w > 100  and h < 80 and h > 10:
                    tempContour.append(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255),2)
   # print( "Total Contours = "+str(len(contours)))
   # print( "Target Contours = " + str(len(tempContour)))
    
# If we didnt find the needed contours return false
    if len(tempContour) == 0 :
        return {'flag' : False , 'img' : Original , 'x' : None, 'y' : None }

        
    # Sort By Arc Lenght and draw the Longest Contour
    cntsSorted = sorted(tempContour, key=lambda x: cv2.arcLength(x,True))
    (x, y, w, h) = cv2.boundingRect(cntsSorted[-1])
    #Draw the Best Contour 
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 10, 255),2)
    # then apply fitline() function
    [vx,vy,x,y] = cv2.fitLine(cntsSorted[-1],cv2.DIST_L2,0,0.01,0.01)
    # Now find two extreme points on the line to draw line
    lefty = int((-x*vy/vx) + y)
    righty = int(((gray.shape[1]-x)*vy/vx)+y)


    #Finally draw the line
    cv2.line(frame,(gray.shape[1]-1,righty),(0,lefty),(0,0,0),1)

# ______________Crop __________________    
    ######################################## 
    # Take the Higher Value 
    if righty < lefty : 
        offset = righty-5
    else :
        offset = lefty-5
         
    if offset <0   : 
        offset = 0 
        
    # check if the horizontal line is too low
    #In the bottom 25% of the img
    #print((gray.shape[0])- (gray.shape[0]/10))
    if offset > (gray.shape[0])- (gray.shape[0]*2.5/10) :
        offset = (gray.shape[0])- (gray.shape[0]*2.5/10)
        return {'flag' : False , 'img' : Original , 'x' : None, 'y' : None }

    
    offset = int(offset)
    CheckUnderLine = gray[offset: , : ]
    #cv2.imshow('Cropped',CheckUnderLine)
    ###############################################################################################################################
    
    edges22 = cv2.Canny(CheckUnderLine,10,20,apertureSize = 3)
    lines = cv2.HoughLinesP(image=edges22,rho=1,theta=np.pi/180, threshold=50, minLineLength=100,maxLineGap=90)
   # Check if we didnt find any lines in the cropped img
    if lines is None:
        return {'flag' : False , 'img' : Original , 'x' : None, 'y' : None }

    
    a,b,c = lines.shape
    #print("A = " + str(a))
    Vert_Line = np.zeros((2,4) , dtype='i')
    TempLine = np.zeros((1,4))
    RefLine = np.zeros((1,4))
    CountVert1,CountVert2,count  = 0 ,0 ,0     # To count Vertical Lines to get Avg
    Width = 0
    for i in range(a):
        # Search For the Vertical Lines 
        #cv2.line(frame, (lines[i][0][0] , lines[i][0][1]), (lines[i][0][2] , lines[i][0][3]), (0, 255, 255), 3, cv2.LINE_AA)

        if abs(lines[i][0][1] - lines[i][0][3]) > 5 and abs(lines[i][0][0] - lines[i][0][2]) < 60:
            count = count+1
           # cv2.line(frame, (lines[i][0][0] , lines[i][0][1]), (lines[i][0][2] , lines[i][0][3]), (0, 255, 255), 3, cv2.LINE_AA)
            TempLine = BotPointThenUpper(lines[i][0])
              # First Line 
            if Vert_Line[0][0] == 0 and  Vert_Line[0][1] == 0:
                #To get The bottom Point
                 #if abs(TempLine[2] - (offset +5)) > 100:
                   # cv2.line(frame, (lines[i][0][0] , lines[i][0][1]), (lines[i][0][2] , lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
                   # continue
                 Vert_Line[0] = TempLine
                 RefLine = TempLine
                 CountVert1 = CountVert1 + 1
        # Check the variance of the Horizontal axis to get the other vertical line
            elif abs(TempLine[0] - RefLine[0]) > 50:
                Vert_Line[1] =  Vert_Line[1] + TempLine
                CountVert2 = CountVert2 + 1
              #  cv2.line(frame, (lines[i][0][0] , lines[i][0][1]), (lines[i][0][2] , lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
            else :    
                Vert_Line[0] =  Vert_Line[0] +TempLine
                CountVert1 = CountVert1 + 1
               # cv2.line(frame, (lines[i][0][0] , lines[i][0][1]), (lines[i][0][2] , lines[i][0][3]), (0, 255, 255), 3, cv2.LINE_AA)
             
    if CountVert1 == 0 or CountVert2 == 0:
      #  print("Something is Wronge")
        return {'flag' : False , 'img' : Original , 'x' : None, 'y' : None }
#Create The best two Vertical lines based on the avg
    Vert_Line[0] = (Vert_Line[0]/CountVert1)            
    Vert_Line[1] = (Vert_Line[1]/CountVert2)
# 1/5 of the total Width
         
    # Vertical Line { 1 }   with Contour Line
    p1_V1 = np.array([ Vert_Line[0][0]  , Vert_Line[0][1]+offset ])
    o1_V1  = np.array([Vert_Line[0][2] ,Vert_Line[0][3]+offset])
    p2_V1 = np.array([gray.shape[1]-1,righty])
    o2_V1 = np.array([0,lefty])
    flag_V1,r_V1 = TwoLineIntersection(o1_V1,p1_V1,p2_V1,o2_V1)
    if (flag_V1 == True) :
       if math.sqrt( ((r_V1[0]-x)**2)+((r_V1[1]-y)**2) ) < 300: 
             cv2.line(frame, (Vert_Line[0][0], Vert_Line[0][1]+offset),(r_V1[0],r_V1[1]), (100, 100, 255), 3, cv2.LINE_AA)
            
       else : 
            cv2.line(frame, (Vert_Line[0][0], Vert_Line[0][1]+offset),(r_V1[0],r_V1[1]), (0, 0, 100), 3, cv2.LINE_AA)
            #print("V1 = " + str(math.sqrt( ((r_V1[0]-x)**2)+((r_V1[1]-y)**2) )))
    cv2.line(frame, (x,y),(r_V1[0],r_V1[1]), (0, 200, 100), 3, cv2.LINE_AA)
   # cv2.imshow("Last2121212121" , frame)
     
        # Vertical Line { 2 }   with Contour Line      
    p1_V2 = np.array([ Vert_Line[1][0]  , Vert_Line[1][1]+offset ])   
    o1_V2  = np.array([Vert_Line[1][2] ,Vert_Line[1][3]+offset])
    p2_V2 = np.array([gray.shape[1]-1,righty])
    o2_V2 = np.array([0,lefty])
    flag_V2,r_V2 = TwoLineIntersection(o1_V2,p1_V2,p2_V2,o2_V2)
    if (flag_V2 == True) :
        if math.sqrt( ((r_V2[0]-x)**2)+((r_V2[1]-y)**2) ) <300 : 
            cv2.line(frame, (Vert_Line[1][0], Vert_Line[1][1]+offset),(r_V2[0],r_V2[1]), (100, 100, 255), 3, cv2.LINE_AA)
           # cv2.imshow("Last2121212121" , frame)
        else : 
            cv2.line(frame, (Vert_Line[1][0], Vert_Line[1][1]+offset),(r_V2[0],r_V2[1]), (0, 0, 100), 3, cv2.LINE_AA)
           # print("V2 = " + str(math.sqrt( ((r_V2[0]-x)**2)+((r_V2[1]-y)**2) )))

    
    if flag_V1 == True and flag_V2 == True : 
        cv2.line(frame, (r_V1[0],r_V1[1]),(r_V2[0],r_V2[1]), (100, 100, 255), 3, cv2.LINE_AA)
        Width = math.ceil(abs(r_V1[0] - r_V2[0]) *  .2 )   
        if r_V1[0] >  r_V2[0] : 
             cv2.line(frame, (r_V2[0], math.ceil((r_V2[1] +  Vert_Line[1][1]+offset )/2)), (r_V2[0] + Width, math.ceil((r_V2[1] +  Vert_Line[1][1]+offset )/2)), (255, 255, 255), 3, cv2.LINE_AA)
            # return True,frame, [r_V2[0] + Width , math.ceil(abs(r_V2[1] +  Vert_Line[1][1]+offset )/2]
             return {'flag' : True , 'img' : frame , 'x' : r_V2[0] + Width  , 'y' : math.ceil((r_V2[1] +  Vert_Line[1][1]+offset )/2) }

        else : 
             cv2.line(frame, (r_V1[0], math.ceil((r_V1[1] +  Vert_Line[0][1]+offset )/2)), (r_V1[0] + Width, math.ceil((r_V1[1] +  Vert_Line[0][1]+offset )/2)), (255, 255, 255), 3, cv2.LINE_AA)
             #return True,frame, [r_V1[0] + Width , math.ceil(abs(r_V1[1] +  Vert_Line[0][1]+offset )/2]
             return {'flag' : True , 'img' : frame , 'x' : r_V1[0] + Width  , 'y' : math.ceil((r_V1[1] +  Vert_Line[0][1]+offset )/2) }
    
    return {'flag' : True , 'img' : Original , 'x' : None  , 'y' :None }

             

    
   # return {'flag' : True , 'img' : frame , 'x' : [r_V2[0] + Width  , 'y' : math.ceil(abs(r_V2[1] +  Vert_Line[1][1]+offset )/2 }
    #cv2.line(frame, (Vert_Line[0][0], Vert_Line[0][1]+offset), (Vert_Line[0][2], Vert_Line[0][3]+offset), (0, 0, 255), 3, cv2.LINE_AA)
    #cv2.line(frame, (Vert_Line[1][0], Vert_Line[1][1]+offset), (Vert_Line[1][2], Vert_Line[1][3]+offset), (0, 0, 255), 3, cv2.LINE_AA)
    # 1 Horizontal Line
    #cv2.line(frame, (Vert_Line[1][0], righty), (Vert_Line[0][0], righty), (0, 0, 255), 3, cv2.LINE_AA)
 
###############################################################################################################################
 

# 250 86 65 74
x = 1
for NIMG in range(x,x+500):
    start_time = time.time()
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print("E:\Courses\OpenCV Learning\Test Cases\\"+ str(NIMG) + ".jpg" )
    frame = cv2.imread("E:\Courses\OpenCV Learning\Test Cases\\"+ str(NIMG) + ".jpg" ,-1)
    cv2.imshow("Img" , frame)
    result =   DetectGate(frame)
    cv2.imshow("Result" , result['img'])
    print(result['flag'])
    print("--- %s seconds ---" % (time.time() - start_time))
    time.sleep(1)
   #  key = cv2.waitKey(0) & 0xFF

   #  if key == ord('s'):
   #     cv2.imwrite("NewImg.jpg",new_img)
   # if key == ord('q'):
   #       break
    cv2.destroyAllWindows()

cv2.destroyAllWindows()


