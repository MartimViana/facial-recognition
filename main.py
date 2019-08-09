## Imports
import numpy as np
import cv2

#########################################################################################
## Global variables
imgFilePath = "images"
rectColor = (255, 0, 0) # Is in BGR color scheme
rectStroke = 2
cascadeClassifierFile = 'haarcascade_frontalface_alt2.xml'
#########################################################################################

## Code
face_cascade = cv2.CascadeClassifier('cascades/data/' + cascadeClassifierFile)

capture = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    tempIncr = 0

    for(x, y, w, h) in faces:
        print(x, y, w, h)       # print values from face

        # roi: region of interest
        # Save image
        roi_gray = gray[y:y+h, x:x+w]   # Only grayscale frame
        roi_color = frame[y:y+h, x:x+w] # Frame with color

        # Recognize region of interest
        # A deep learned model could be used to predict things here (keras, tensorflow or pytorch)
        # This method doesn't work 100%, but it works!

        
        cv2.imwrite(str(tempIncr)+'.png', roi_gray)
        tempIncr = tempIncr + 1

        # Draw rectangle on region of interest
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), rectColor, rectStroke)   # Draw rectangle

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindws()