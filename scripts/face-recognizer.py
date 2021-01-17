import os

import cv2
import face_recognition
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, '../datasets/face')

name = 'Anne2'
cap = cv2.VideoCapture('../src/' + name + '.mp4')
cap.set(3, 1080)
cap.set(4, 720)
labels = {}

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(image_dir + '\\trainer-faces.yml')

with open(image_dir + '\\labels-faces.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

scale = 0.20
scaleV = 5

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break

    frame_scale = cv2.resize(frame, (0, 0), None, scale, scale)
    frame_scale = cv2.cvtColor(frame_scale, cv2.COLOR_RGB2GRAY)
    faces = face_recognition.face_locations(frame_scale)

    for face in faces:
        y1, x2, y2, x1 = face
        y1, x2, y2, x1 = y1 * scaleV, x2 * scaleV, y2 * scaleV, x1 * scaleV
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gray_face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        id_, conf = recognizer.predict(gray_face)

        name = labels[id_]
        text = str(name) + '<=>' + str(round(conf, 2))
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # cv2.imshow('gray frame', frame_scale)
    cv2.imshow('rgb frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
