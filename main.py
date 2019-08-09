## Imports
import numpy as np
import cv2
import pickle

#########################################################################################
## Global variables
imgFilePath = "images"
rectColor = (255, 0, 0) # Is in BGR color scheme
rectStroke = 1
cascadeClassifierFile = 'haarcascade_frontalface_alt2.xml'
font = cv2.FONT_HERSHEY_PLAIN
colorFont = (255, 255, 255)
fontStroke = 2
fontSize = 1
MIN_CONF = 45
MAX_CONF = 75
#########################################################################################

## Code
face_cascade = cv2.CascadeClassifier('cascades/data/' + cascadeClassifierFile)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
tempLabels = {}
with open("labels.pickle", "rb") as f:
    tempLabels = pickle.load(f)
    labels = {v:k for k,v in tempLabels.items()}

capture = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    tempIncr = 0

    for(x, y, w, h) in faces:
        #print(x, y, w, h)       # print values from face

        # roi: region of interest
        # Save image
        roi_gray = gray[y:y+h, x:x+w]   # Only grayscale frame
        roi_color = frame[y:y+h, x:x+w] # Frame with color

        # Recognize region of interest
        # A deep learned model could be used to predict things here (keras, tensorflow or pytorch)
        # This method doesn't work 100%, but it works!
        id, conf = recognizer.predict(roi_gray)
        if conf >= MIN_CONF and conf  <= MAX_CONF:
            name = labels[id]
            cv2.putText(frame, name + ", " + str(conf), (x, y+h), font, fontSize, colorFont, fontStroke, cv2.LINE_AA)

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
cv2.destroyAllWindows()
