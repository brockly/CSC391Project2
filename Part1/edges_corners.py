'''
Alex Brockman
Part 1 Implementation
While writing the report, make sure to change every variable as possible and
analyze each picture one by one in edge AND corner detection
'''

import cv2

#use camera to capture video
video = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    # Perform operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corner = cv2.cornerHarris(gray, 2, 5, 0.04) #default: 2, 3, 0.02
    canny = cv2.Canny(gray, 175, 175) #default: gray, 200, 200

    # dilate the edges and corners to view in color video
    corner = cv2.dilate(corner, None)
    canny = cv2.dilate(canny, None)

    frame[corner > 0.001 * corner.max()] = [0, 0, 255] #changes size of dots in frame
    frame[canny > 0.001 * canny.max()] = [0, 0, 255] #changes size of dots in frame

    #display the frame
    cv2.imshow('Edges', frame)

    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()

