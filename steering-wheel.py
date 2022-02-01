import mediapipe as mp
from mediapipe.python.solutions import hands
import cv2
import imutils
import numpy as np
import time
from pynput.keyboard import Key, Controller
import pynput
# keyboard = Controller()

import pydirectinput

# Handling our video object, zero value mean camera number 1
cap = cv2.VideoCapture(1)

keyboard = Controller()

# const colors
BLUE=(255,0,0)
RED=(0,0,255)
YELLOW=(51,51,51)

# Time Delay
w_delay= 100
d_delay=200
a_delay=200
s_delay=300





# Handling Hands object from mediapipe libairy
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()
    #Handling detection yellow color in left and right
    frame=cv2.flip(frame,1)
    frame = imutils.resize(frame,height=400, width=600)
    
    rectangle = cv2.rectangle(frame, (10,100),(250,320),YELLOW,1)
    cv2.putText(rectangle, "LEFT", (100, 130 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
   
    rectangle = cv2.rectangle(frame, (590,100),(350,320),YELLOW,1)
    cv2.putText(rectangle, "RIGHT", (430, 130 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # HVS Color
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    for cnt in contours:
        (x,y,w,h)= cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
        if x<250:
            if (time.time()-a_delay) >= 1:
                a_delay = time.time()
                pydirectinput.keyUp('d')
                pydirectinput.keyDown('a')
                
                key = True
            
        if x>303:
            if (time.time()-d_delay) >= 1:
                d_delay = time.time()
                pydirectinput.keyUp('a')
                pydirectinput.keyDown('d')
                
                key = True
        break   
    
    
    # Coverting into RGB, becouse class {hands} only uses RGB images
    frameRGP = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
    results= hands.process(frameRGP)
    lmList=[]
    lmList2=[]
    # Check if we have multiple hands or not
    if results.multi_hand_landmarks:
        # Draw Points in Hands {21 LandMarks}
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                if len(lmList)<21:
                    lmList.append([id , cx , cy])
                else:
                    lmList2.append([id , cx , cy])
                # Draw Circle on every handmark
                # cv2.circle(frame,(cx, cy), 5, (255,0,255), cv2.FILLED)
            
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        
    if len(lmList) !=0: 
        
        if abs(lmList[20][2]-lmList[4][2]) >=100 :     
            s_delay = time.time() 
            pydirectinput.keyUp('w')
            pydirectinput.keyDown('s')
                

        for i in range(2,5):   
            x1,y1 = lmList[i][1],lmList[i][2]
            cv2.circle(frame,(x1,y1),5,(255,0,255),cv2.FILLED)
        
        if len(lmList2) !=0:
            
            for i in range(2,5):      
                x1,y1 = lmList2[i][1],lmList2[i][2]
                cv2.circle(frame,(x1,y1),5,(255,0,255),cv2.FILLED)
            if abs(lmList2[20][2]-lmList2[4][2]) >=100:
                pydirectinput.keyUp('s')
                pydirectinput.keyDown('w')
                w_delay = time.time()
            

    # Handling Release buttons
    if (time.time()-s_delay)>2: 
        s_delay = time.time()
        pydirectinput.keyUp('s')
    if (time.time()-d_delay)>0.2:
        pydirectinput.keyUp('d')
    if (time.time()-a_delay)>0.2:
        pydirectinput.keyUp('a')
    if (time.time()-w_delay)>2:
        w_delay = time.time()
        pydirectinput.keyUp('w')

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
        
    cv2.imshow("Steering Wheel", frame)
    cv2.waitKey(1)
