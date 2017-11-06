import fcntl
import time
import cv2

img = "./mem_image/cap.png"

while(True):
    img_l = open(img,'r')
    fcntl.flock(img_l,fcntl.LOCK_EX)
    print("got lock")
    #cv2.imwrite(img,gray)
    fcntl.flock(img_l,fcntl.LOCK_UN)
    img_l.close()
    


    # Display the resulting frame
    #cv2.imshow('frame',gray)
    #cv2.waitKey(1)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

