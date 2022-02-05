import cv2
import numpy as np

def nothing(n):
    pass

# Contours w/ greatest number of points
# TODO max by area
def biggestContourI(contours):
    maxVal = 0
    maxI = -1
    for i in range(0, len(contours)):
        if len(contours[i]) > maxVal:
            cs = contours[i]
            maxVal = len(contours[i])
            maxI = i
    return maxI

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

iLowH = 40;
iHighH = 90;
iLowS = 47; 
iHighS = 255;
iLowV = 49;
iHighV = 255;
abc = 0

cv2.namedWindow('Control')
cv2.createTrackbar("LowH", "Control", iLowH, 180, nothing);
cv2.createTrackbar("HighH", "Control", iHighH, 180, nothing);
cv2.createTrackbar("LowS", "Control", iLowS, 255, nothing);
cv2.createTrackbar("HighS", "Control", iHighS, 255, nothing);
cv2.createTrackbar("LowV", "Control", iLowV, 255, nothing);
cv2.createTrackbar("HighV", "Control", iHighV, 255, nothing);
#cam = cv2.VideoCapture('C:\Users\Jeremy\AppData\Local\Programs\Python\Python37-32\Vision Code\ez.jpg')
cam = cv2.imread("C:\\ision2022-Competition\\HubImgFRC\\NearLaunchpad13ft6in.png")

while True:
    #ret_val, img = cam.read()
    
    img = cv2.imread("C:\\Vision2022-Competition\\HubImgFRC\\NearLaunchpad13ft6in.png")
    ret_val = 0
    
    lh = cv2.getTrackbarPos('LowH', 'Control')
    ls = cv2.getTrackbarPos('LowS', 'Control')
    lv = cv2.getTrackbarPos('LowV', 'Control')
    hh = cv2.getTrackbarPos('HighH', 'Control')
    hs = cv2.getTrackbarPos('HighS', 'Control')
    hv = cv2.getTrackbarPos('HighV', 'Control')

    lower = np.array([lh, ls, lv], dtype = "uint8")
    higher = np.array([hh, hs, hv], dtype = "uint8")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    flt = cv2.inRange(hsv, lower, higher);
    
    contours0, hierarchy = cv2.findContours(flt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Only draw the biggest one
    bc = biggestContourI(contours0)
    #print("bc " + str(bc))
    #print("contours " + str(contours0))
    if bc != -1 :
        cv2.drawContours(img,contours0, bc, (220,0,255), 3)

    lower_green = np.array([55,90,50])
    upper_green = np.array([90,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(img,img, mask= mask)
    
    cv2.imshow('my webcam', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('flt', flt)
    cv2.imshow('res',res)

    #cv2.imwrite('C:\Users\Jeremy\AppData\Local\Programs\Python\Python37-32\Vision Code\ez.jpg', res)
    
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()

# Good for test 1 Low: 22, 126, 0. High: 55, 255, 255.
