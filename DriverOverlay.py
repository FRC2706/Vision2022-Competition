import cv2
import numpy as np
import math
from VisionUtilities import * 
from VisionConstants import *
from DistanceFunctions import *
from networktables import NetworkTablesInstance
from networktables.util import ntproperty

try:
    from PrintPublisher import *
except ImportError:
    from NetworkTablePublisher import *

# A simple Driver Overlay.  Requires 
#   final center which is the center of the object being targeted
#   Yaw to Target, the Yaw value to the target
#   The Distance to the target

def DriverOverlay(frame, final_center,YawToTarget, distance):
  
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
  
    #Now read Distance and Yaw from Network tables
    if final_center != -99:
        cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), white, 2)
        if (YawToTarget >= -2 and YawToTarget <= 2):
           colour = green
        if ((YawToTarget >= -6 and YawToTarget < -2) or (YawToTarget > 2 and YawToTarget <= 6)):  
           colour = yellow
        if ((YawToTarget < -6 or YawToTarget > 6)):  
           colour = red
        cv2.line(image, (round(final_center), screenHeight), (round(final_center), 0), colour, 2)

    if YawToTarget != -99:        
        cv2.putText(image, "TargetYaw: " + str(YawToTarget), (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)

    if distance != -1:    
        cv2.putText(image, "Distance: " + str(round((distance/12),2)), (20, 600), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)

  
    # Shows the contours overlayed on the original video
    return image

