import cv2
import numpy as np
import math
from Gestures import *
from AsyncControlRepeater import AsyncControlRepeater
T_RIGHT_CLICK = 50  # max cnt is larger than next larger cnt area with this value => more value means more possible to be recognized over left click and no shape
T_LEFT_CLICK = 20   # hull area is larger than cnt area with at least this value => less value means more possible to be recognized over no shape
T_MOVING = 2        # less value means more possible to be moving, [from 1 to 4]

T_MIN_DISTANCE = 15 # less value means more accuracy for move but also more noise in others ==> is this value made very minimum, consider using more value in T_MOVING
class GestureRecognizer:
    def __init__(self):
        pass

    def fromFeatures(self, numDefects, lengthRatio, hullCntRatio, maxTwoCntRatio):
        print('defects=',numDefects)
        print('lengthRatio=',lengthRatio)
        print('hullCntRatio=',hullCntRatio)
        print('maxTwoCntRatio=',maxTwoCntRatio)
        if numDefects >= 2:
            return PALM
        elif lengthRatio > 2:
            return KNIFE
        elif maxTwoCntRatio < 250 :
            return ZERO
        elif lengthRatio < 1.45 and lengthRatio > 0.8 and hullCntRatio > 0.9 and hullCntRatio < 1.1: # square like
            return FIST
        else:
            return NO_GST

    def recognize(self, roi, handMask):
        try:
            #handMask = cv2.erode(handMask,np.ones((3,3)),iterations=3)
            #handMask = cv2.dilate(handMask,np.ones((3,3)),iterations=3)
            #handMask = roi * handMask
            hand = handMask
            

            hand = self.__preProcessing__(hand)
            contours = self.__findContoursSorted__(hand, roi)

            maxCnt = contours[0]
            #x,y,w,h = cv2.boundingRect(maxCnt)
            #hand = hand[y:y+h, x:x+w]

            #contours = self.__findContoursSorted__(hand, roi)
            maxCnt = contours[0]

            maxCntArea = cv2.contourArea(maxCnt)
            maxCnt2Area = cv2.contourArea(contours[1]) if len(contours) >=2 else 0.0001
            f_maxTwoCntRatio = (maxCntArea - maxCnt2Area) / maxCnt2Area
            
            hull = cv2.convexHull(maxCnt)
            areahull = cv2.contourArea(hull)
            #f_hullCntRatio = (areahull - maxCntArea) / maxCntArea
            f_hullCntRatio = areahull / maxCntArea


            x,y,w,h = cv2.boundingRect(maxCnt)
            cv2.rectangle(hand,(x,y),(x+w,y+h),(100),5)  # * [debug]
            f_lengthRatio =  h / float(w)

            f_defects = self.__findDefects__(roi, maxCnt)
            print(f_defects)
            cv2.imshow('hand_gestureRecog',hand)
            return self.fromFeatures(f_defects, f_lengthRatio, f_hullCntRatio, f_maxTwoCntRatio), self.__contourCenter__(maxCnt)
        except Exception as e:
            print(e)
            return NO_GST, (0,0)

    def __preProcessing__(self, mask):
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 4)
        mask = cv2.GaussianBlur(mask,(5,5),100)
        return mask

    def __contourCenter__(self, cnt):
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def __findContoursSorted__(self, hand, roi):
        _,contours,hierarchy= cv2.findContours(hand,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
        cv2.drawContours(roi, contours,-1, color=(255,0,0))
        cv2.drawContours(hand, contours,-1, color=(100))
        return contours

    def __findDefects__(self, roi, cnt):
        #find the defects in convex hull with respect to hand
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        f_numDefects = 0
        
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
            if angle <= 90 and d > T_MIN_DISTANCE:
                f_numDefects += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            else:
                cv2.circle(roi, far, 3, [0,0,255], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)

        return f_numDefects

   

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

from GameController import GameController
from ConfigReader import ConfigReader
if __name__ == "__main__":
    gr = GestureRecognizer()
    cap = cv2.VideoCapture(0)
    #gc = GameController(config=ConfigReader.default())
    gc = AsyncControlRepeater(GameController(config=ConfigReader.default()), maxBufferSize=1).start()
    while(True):
        _ , _frame = cap.read()
        _frame = cv2.flip(_frame,1)
        #_frame = adjust_gamma(_frame, 1.0)

        #define region of interest
        start = 50
        end = 350
        _roi=_frame[start:end, start:end]
        cv2.rectangle(_frame,(start,start),(end,end),(0,255,0),0)

        _hsv = cv2.cvtColor(_roi, cv2.COLOR_BGR2HSV)
        #print(_hsv[:50][0:50])
        # define range of skin color in HSV
        _lower_skin = np.array([5, 5, 80], dtype=np.uint8)
        _upper_skin = np.array([60, 250, 250], dtype=np.uint8) 

        _mask = cv2.inRange(_hsv, _lower_skin, _upper_skin)

        _gesture, _center = gr.recognize(_roi, _mask)
        print(_gesture, _center)

        _font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(_frame, _gesture, (0,50), _font, 2, (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow('frame',_frame)
        delay = np.random.randint(20, 25, 1)[0] * 10 # in seconds
        print('delay', delay)
        k = cv2.waitKey(1) & 0xFF
        delay = delay / 1000 # in ms

        #gc.control(_gesture, _center, dt=delay)
        while(not gc.addGstRecognition(_gesture, _center)):
            pass
        if k == 65:
            break

    cv2.destroyAllWindows()
    cap.release()