import face_recognition
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

from keras.preprocessing import image

model = load_model('../models/face_low_val_loss.h5')
model_gender = load_model('../models/gender.model')

classes_gender = ['WOMAN', 'MAN']
classes_face = ['Anne', 'Ben', 'Billie', 'Jack']

# Loading the cascades
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture('../src/anne.mp4')

scale = 0.20
scaleV = 5

# Doing some Face Recognition with the webcam
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    if frame is None:
        break

    frame_scale = cv2.resize(frame, (0, 0), None, scale, scale)
    frame_scale = cv2.cvtColor(frame_scale, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(frame_scale)

    for face in faces:
        y1, x2, y2, x1 = face
        gray_face = frame_scale[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (224, 224), interpolation=cv2.INTER_LINEAR)
        gray_face = gray_face / 255.

        img_array = img_to_array(gray_face)
        img_array = img_array.reshape(1, 224, 224, 3)

        pred = model.predict(img_array, 1, verbose=0)
        conf, index = np.max(pred, axis=1), np.argmax(pred, axis=1)
        rounded = str(np.round(conf * 100, 2))

        print(pred)

        if conf > 0.55:
            name = rounded + ' ' + str(classes_face[int(index)])
        else:
            name = rounded + 'Unknown'

        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # preprocessing for gender detection model
        """face_crop = cv2.resize(face, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        pred_gender = model_gender.predict(face_crop)[0]
        gender = str(classes_gender[pred_gender.argmax()]) + ' / ' + \
                 str(round(max(pred_gender) * 100, 2)) if pred_gender.max() > 0.9 else 'unknown'
        cv2.putText(frame, gender, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)"""

    cv2.imshow('Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
