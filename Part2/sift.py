'''
Alex Brockman
Part 2 Implementation
Experiment with changing the sift parameters, should not take too long
'''

import cv2
import numpy as np

#read in image
img = cv2.imread('float_boat_40%.jpg')

#convert to gray and apply sift function
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create() #nfeatures=n to specify

#create keypoints and draw them on the picture
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(img,kp, None, flags=cv2.DrawMatchesFlags_DEFAULT)

#write the finished result to file
cv2.imwrite('sift_keypoints1_smaller.jpg',img)