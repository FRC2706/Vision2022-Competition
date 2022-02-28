#THIS FILE CONSISTS OF VISION CONSTANTS, EXPECTED TO BE USED EVERY YEAR
import math
import numpy as np

# Field of View (FOV) of the microsoft camera (68.5 is camera spec)
# Lifecam 3000 from datasheet
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf

diagonalView = math.radians(68.5)

#print("Diagonal View:" + str(diagonalView))

# 4:3 aspect ratio
horizontalAspect = 4
verticalAspect = 3

# Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
# Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView / 2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView / 2) * (verticalAspect / diagonalAspect)) * 2

# MAY CHANGE IN FUTURE YEARS! This is the aspect ratio used in 2022
image_width = 1280 # 4  
image_height = 720 # 3  

H_FOCAL_LENGTH = image_width / (2 * math.tan((horizontalView / 2)))
V_FOCAL_LENGTH = image_height / (2 * math.tan((verticalView / 2)))

#CARGO_HEIGHT is actual height (for cargo height in feet)   
CARGO_BALL_HEIGHT = 0.791667

#image height is the y resolution calculated from image size
#223 is the pixel height of a a ball found at a measured distance (which is 4 feet away)
#65 is the pixel height of a scale image 6 feet away
KNOWN_CARGO_PIXEL_HEIGHT = 223
KNOWN_CARGO_DISTANCE = 4

# Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
# H_FOCAL_LENGTH = image_width / (2 * math.tan((horizontalView / 2)))
# V_FOCAL_LENGTH = image_height / (2 * math.tan((verticalView / 2)))
# blurs have to be odd
green_blur = 1
orange_blur = 27
yellow_blur = 1
red_blur = 1
blue_blur = 1

# define colors
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (252, 252, 3)
white = (255, 255, 255)
yellow = (0, 255, 255)
orange = (60, 255, 255)

# define range of green of retroreflective tape in HSV
lower_green = np.array([55, 85, 146])
upper_green = np.array([94, 255, 255])

# define range of green of retroreflective tape in HSV
#lower_green = np.array([23, 50, 35])
#upper_green = np.array([85, 255, 255])

lower_yellow = np.array([10, 150, 65]) # was 14, 150, 150
upper_yellow = np.array([30, 255, 255])

# masks for red and blue cargo (HSV)
lower_red = np.array([138,106,123])
upper_red = np.array([180,255,255])

lower_blue = np.array([64,127,116]) 
upper_blue = np.array([115,213,255]) 

blingColour = 0
