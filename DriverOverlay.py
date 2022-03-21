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


# A simple calculation that scales an X position value based on the X point of the
# current camera and an alternate Camera
def calculateResolutionDifferential(final_center, driverCamWidth, AltCamWidth):
    #calculate final_center for simple resolution difference
    scaledCenter = (final_center * (driverCamWidth/AltCamWidth))
    return scaledCenter

# A simple Driver Overlay.  Requires 
#   frame:  The frame (image) which will be overlayed
#   cameraFOV:  The Field of view of the current camera type
#   final center which is the center of the object being targeted which may have been
#                calcualted using a different field of view.
#   Yaw to Target, the Yaw value to the target calculated using the alternate camera with
#           an alternate field of view
#   The Distance to the target (for display purposes)

def DriverOverlay(frame, cameraFOV, final_center,YawToTarget, distance):
  
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()

    #final_center = -99

    #print("harcoded focalLength:",H_FOCAL_LENGTH)

    H_FocalLength, V_FocalLength = calculateFocalLengthsFromInput(cameraFOV, screenWidth, screenHeight)
    #print("calculated focal:",H_FocalLength)
    if (YawToTarget != -99):
      # print("overlay1: ",final_center) 
       final_center = getTargetCenterFromYaw(YawToTarget, centerX, H_FocalLength)
       #print("overlay2: ",final_center)
       #final_center = calculateResolutionDifferential(final_center,screenWidth,screenWidth)
       #print("shape:",screenWidth,screenHeight)
       #print("overlay: ",final_center)

  
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
        cv2.putText(image, "TargetYaw: " + str(YawToTarget), (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)

    if distance != -1:    
        cv2.putText(image, "Distance: " + str(round((distance),2)), (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0,white)

  
    # Shows the contours overlayed on the original video
    return image

