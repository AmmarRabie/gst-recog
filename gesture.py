import cv2
import numpy as np
import math
from scipy import ndimage
cap = cv2.VideoCapture(0)

while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # print(hsv[100,100], "end")
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100)
        

        
    #find contours
        _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
        cv2.drawContours(roi, contours,0, color=(255,0,0))
    
   #find contour of max area(hand)
        #print([cv2.contourArea(x) for x in cnts], 'hiiiii')
        cnt = contours[0]
        cnt2 = contours[1] if len(contours) >=2 else None
        areacnt2 = cv2.contourArea(cnt2) if len(contours) >=2 else None
        #x,y,w,h = cv2.boundingRect(cnt2)
        #cv2.rectangle(mask,(x,y),(x+w,y+h),(200),5)
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

    #make convex hull around hand
        hull = cv2.convexHull(cnt)

        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.intp(box)
        # cv2.drawContours(mask,[box],0,(150),2)

        # (x,y),radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x),int(y))
        # radius = int(radius)
        # print(center,radius)
        # print(mask.shape, mask.dtype)
        # cv2.circle(mask,center,radius,(100),2)
        x,y,w,h = cv2.boundingRect(cnt)
        focused = mask[y:y+h,x:x+w]
        cv2.rectangle(mask,(x,y),(x+w,y+h),(100),5)
        cv2.imshow("focuss",focused)
        f_widthHightRatio =  h / w
        print('height,', h, 'w=', w)
        print(f_widthHightRatio)
        #resized = cv2.resize(focused, (200, 200))
        #cv2.imshow("resized",resized)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
            

         #//l+=1
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l >= 2:
            cv2.putText(frame,'moving',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif f_widthHightRatio > 2 :
            cv2.putText(frame,'pause',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif areacnt2 != None and areacnt / areacnt2 > 20 :
            cv2.putText(frame,'right click',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif f_widthHightRatio < 1.4 and f_widthHightRatio > 0.6 and arearatio < 20:
            cv2.putText(frame,'left click',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame,'NO SHAPE',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
        #cv2.imshow('mroi',mROI)
    except Exception as e:
        print(e)
        pass
        
    import py
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
    




