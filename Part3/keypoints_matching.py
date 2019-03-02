'''
Alex Brockman
Part 3 Implementation
Make sure to compare the use of Harris corners to SIFT Keypoints on several examples
'''

import numpy as np
import cv2

#read images and resize them
img1 = cv2.imread('book_front.jpg', 0)
img2 = cv2.imread('book_side.jpg', 0)
#img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)
#img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

#Use orb detector to grab keypoints and descriptors
#Use BFMatcher to match the two points
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

#Create and compare 10 matches
matches = sorted(matches, key=lambda x: x.distance)
comparing = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], outImg=None, flags=2)

#show final image
cv2.imshow('Comparison Image', comparing)
cv2.imwrite('book_matching.jpg', comparing)
key = cv2.waitKey(0)