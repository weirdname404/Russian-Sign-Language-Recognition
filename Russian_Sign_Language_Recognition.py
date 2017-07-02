# Created by Alexander Lebedev 
# 20.07.2017

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
IMG = cv2.imread('RSL.jpg')

# font = cv2.FONT_HERSHEY_SIMPLEX
# put_text_color = (18,0,255)
# put_text_pos = (60,50)


lower_thresh1 = 129 
upper_thresh1 = 255

PI = math.pi

while (cap.isOpened()):
    
    # Take each frame
    ret, frame = cap.read()

    # Smoothing the input
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Mirror effect
    frame = cv2.flip(frame, 1)
    
    # Setting up output size
    cv2.rectangle(frame,(350,60),(600,300),(255,255,255),2) 
    crop_frame = frame[60:300, 350:600]
    
    # Convert frame to gray color
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0, 48, 80], dtype = "uint8")
    upper_red = np.array([20, 255, 255], dtype = "uint8")

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Subtract the background
    value = (35, 35)
    blurred = cv2.GaussianBlur(gray, value, 0)
    ret, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Extracting contours and finding the largest one
    _,contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # Area contour
    area_of_contour = cv2.contourArea(cnt)
    
    # Straight Bounding Rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    
    # Bounding Rectangle
    cv2.rectangle(crop_frame, (x,y), (x + w, y + h), (250,30,60), 2)

    # Convex Hull 
    hull = cv2.convexHull(cnt)

    drawing = np.zeros(crop_frame.shape, np.uint8)

    # all the points which comprises that object
    cv2.drawContours(drawing,[cnt],0,(255, 0, 255),0)
    cv2.drawContours(drawing,[hull],0,(0, 255, 0),0)
    pixelpoints = np.transpose(np.nonzero(mask))

    # returnPoints - By default, True. Then it returns the coordinates of the hull points. 
    # If False, it returns the indices of contour points corresponding to the hull points.
    hull  = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    count_defects = 0
    
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])


        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 60

        # Drawing a circle
        cv2.circle(crop_frame,far,4,[0,0,255],-1)

        if angle <= 90:
            count_defects += 1
        
        # Drawing hull
        cv2.line(crop_frame,start,end,[0, 255, 0],3)

    # Contour Analysis
    moment = cv2.moments(cnt)   
    perimeter = cv2.arcLength(cnt,True)
    area = cv2.contourArea(cnt)

    # Actually it is not needed (diameter of contour area)
    #equi_diameter = np.sqrt(4*area/np.pi)

    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)

    # draw a circle
    #cv2.circle(crop_frame,center,radius,(255,0,0),2)

    area_of_circle = PI * radius * radius

    hull_test = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull_test)
    solidity = float(area)/hull_area

    # the ratio of width to height of bounding rect of the object
    aspect_ratio = float(w)/h

    rect_area = w*h
    extent = float(area)/rect_area
     
    # Orientation is the angle at which object is directed.
    (x,y),(MA,ma),angle_t = cv2.fitEllipse(cnt)

    # Test
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    ############################################
    # RECOGNITION

    if area_of_circle - area > 33000.0:
           cv2.putText(frame, "No hand", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (50,0,255), 2, cv2.LINE_AA)
    
    elif area_of_circle - area <= 33000.0:
        if 115.0 <= angle_t <= 143.0:
            if 0.96 <= solidity <= 0.99:
                if count_defects <= 1:
                    cv2.putText(frame, "A", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)


        if 173.0 <= angle_t <= 180.0:
            if 0.80 <= solidity <= 0.89:
                if count_defects in range(0, 2):
                    cv2.putText(frame, "BE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)
        elif 0.0 <= angle_t <= 5.0:
            if 0.80 <= solidity <= 0.89:
                if count_defects in range(0, 2):
                    cv2.putText(frame, "BE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)


        if 162.0 <= angle_t <= 180.0:
            if 1 <= count_defects <= 2:
                if 0.86 <= solidity <= 0.95:
                    cv2.putText(frame, "VE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)
                if 0.69 <= solidity <= 0.79:
                    cv2.putText(frame, "VE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)
                
        elif 0.0 <= angle_t <= 8.0:
            if count_defects == 1:
                if 0.86 <= solidity <= 0.95:
                    cv2.putText(frame, "VE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 136.0 <= angle_t <= 170.0:
            if count_defects <= 1:
                if 0.87 <= solidity <= 0.97:
                    cv2.putText(frame, "E", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 110.0 <= angle_t <= 126.0:
            if count_defects <= 1:
                if 0.87 <= solidity <= 0.95:
                    cv2.putText(frame, "ZHE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 0.0 <= angle_t <= 30.0:
            if count_defects == 1:
                if 0.66 <= solidity <= 0.76:
                    cv2.putText(frame, "EE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 0.0 <= angle_t <= 30.0:
            if count_defects == 1:
                if 0.80 <= solidity <= 0.84:
                    cv2.putText(frame, "EE", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 145.0 <= angle_t <= 180.0:
            if 2 <= count_defects <= 3:
                if 0.65 <= solidity <= 0.74:
                    cv2.putText(frame, "EN", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        elif 0.0 <= angle_t <= 16.0:
            if 2 <= count_defects <= 3:
                if 0.61 <= solidity <= 0.75:
                    cv2.putText(frame, "EN", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 150.0 <= angle_t <= 180.0:
            if count_defects > 1:
                if 0.73 <= solidity <= 0.79:
                    cv2.putText(frame, "O", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)


        elif 0.0 <= angle_t <= 30.0:
            if count_defects > 1:
                if 0.73 <= solidity <= 0.89:
                    cv2.putText(frame, "O", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

                       
        if 114.0 <= angle_t <= 137.0:
            if count_defects <= 2:
                if 0.60 <= solidity <= 0.75:
                    cv2.putText(frame, "ES", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 0.0 <= angle_t <= 70.0:
            if count_defects <= 1:
                if 0.70 <= solidity <= 0.75:
                    cv2.putText(frame, "OO", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 110.0 <= angle_t <= 140.0:
            if count_defects <= 1:
                if 0.72 <= solidity <= 0.80:
                    cv2.putText(frame, "EF", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 0.0 <= angle_t <= 15.0:
            if count_defects == 0:
                if 0.93 <= solidity <= 0.96:
                    cv2.putText(frame, "SHA", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 160.0 <= angle_t <= 180.0:
            if count_defects <= 1:
                if 0.55 <= solidity <= 0.70:
                    cv2.putText(frame, "IH*", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 125.0 <= angle_t <= 165.0:
            if count_defects <= 1:
                if 0.75 <= solidity <= 0.86:
                    if area_of_circle - area < 33000.0:
                        cv2.putText(frame, "YOO", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)

        if 125.0 <= angle_t <= 155.0:
            if count_defects <= 2:
                if 0.58 <= solidity <= 0.73:
                   cv2.putText(frame, "m. znak", (60,50), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200, 200, 200), 2, cv2.LINE_AA)




    ############################################
    # SHOWING SCREENS + SOME COMMANDS

    cv2.imshow('Threshold', thresh1)
    cv2.imshow('Contours + Convex Hull', drawing)
    #cv2.imshow('Defects', crop_frame)
    cv2.imshow('Original', frame)
    cv2.imshow('Russian Sign Language Alphabet', IMG)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        print "The solidity is:" , solidity 
        print "The aspect ratio is :", aspect_ratio 
        print "The number of convexity defects are :",count_defects
        print "The extent is :",extent
        print "the angle is:", angle_t
        print "The area of effective circle is", area_of_circle - area, "\n\n"



cap.release()
cv2.destroyAllWindows()