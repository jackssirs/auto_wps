import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
while True:
	# Capture frame-by-frame
	ret, frame = cap.read()
	
	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	cv2.imwrite("./point_b.png",frame)
	exit(0)
	# Display the resulting frame
	#img = cv2.imread("./point_b.png")
	cv2.imshow('frame',frame)
	cv2.waitKey(1)
	#if cv2.waitKey(1) & 0xFF == ord('q'):
	 #   break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
