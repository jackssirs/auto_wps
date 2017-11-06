#!/usr/bin/env /usr/bin/python2.7
import fcntl
import numpy as np
import cv2
import os
import time

cap = cv2.VideoCapture(0)
img = "./mem_image/cap.png"
lockfile = "./mem_image/lock"
oncefile = "./mem_image/once"

while(True):
    #if lockfile exists , we can save.
    if os.path.exists(lockfile):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(gray)
    
        cv2.imwrite(img,gray)
        
        #rm lockfile after we save the image.  
        os.system("rm "+lockfile)
        if os.path.exists(oncefile):
            os.system("rm "+oncefile)
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
