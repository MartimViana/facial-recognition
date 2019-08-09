## Imports
import cv2
import numpy as np
import os
from PIL import Image
import pickle

#########################################################################################
## Global Variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   #current file path
image_dir = os.path.join(BASE_DIR, "images")
cascadeClassifierFile = 'haarcascade_frontalface_alt2.xml'


#########################################################################################

## Code
face_cascade = cv2.CascadeClassifier('cascades/data/' + cascadeClassifierFile)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Searches for images
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)

            # Get name of folder to give label
            # In this case, the label assigned is the name of the folder.
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
            #print(label, path)
            
            # Get training id
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id = label_ids[label]
            #print(label_ids)

            # Convert training image to matrix
            pil_image = Image.open(path).convert("L")   # convert to grayscale
            image_array = np.array(pil_image, "uint8")  # convert image to matrix
            # Know region of interest in image
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)


#print(y_labels)
#print(x_train)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
