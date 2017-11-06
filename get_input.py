import numpy as np
import cv2
import time
import datetime

datadir = "data/"

#status = "0_order/" #order image
#flag = "a"

#status = "1_prewps/" #pre-open-wps image
#flag = "b"

#status = "2_wpsopen/" #open-wps image
#flag = "c"

#status = "3_docend/" #end page of doc  image
#flag = "d"

status = "4_bug/" #bug image
flag = "e"

duration = 0.01 #get image of 0.1 second duration

ISOTIMEFORMAT='%Y%m%d_%H%M%S_'




def get_time():
    return str(time.strftime(ISOTIMEFORMAT))+str(datetime.datetime.now().microsecond)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print (gray)
    #print (np.shape(gray))
    print(datadir+status+flag+"_"+get_time()+".png")
    cv2.imwrite(datadir+status+flag+"_"+get_time()+".png",gray)
    time.sleep(duration)

    # Display the resulting frame
    #cv2.imshow('frame',gray)
    #cv2.waitKey(1)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
