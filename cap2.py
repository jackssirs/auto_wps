import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("./mem_image/cap.png",gray)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
