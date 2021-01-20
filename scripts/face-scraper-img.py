import glob
import os

import cv2
import face_recognition
import numpy as np
from PIL import Image

name = 'selena gomez'
path = '../datasets/face/train/' + name + '/'
os.mkdir(path)

images = glob.glob('../src/' + name + '/*')

scaleV = 5
i = 0

for image in images:
    image = Image.open(image)
    image = np.asarray(image)

    frame_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_recognition.face_locations(frame_scale)

    for face in faces:
        i += 1
        y1, x2, y2, x1 = face
        filename = path + str(i) + '.jpg'
        gray_face = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        gray_face = cv2.resize(gray_face, (224, 224))
        cv2.imwrite(filename, gray_face)

    cv2.imshow('rgb frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q') or i > 500:
        break

cv2.destroyAllWindows()
print("Collecting Samples Complete")
