import os
import pickle

from PIL import Image
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, '../datasets/face')

y_labels = []
x_train = []
current_id = 0
label_ids = {}

recognizer = cv2.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()

            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, 'uint8')

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            y_labels.append(id_)
            x_train.append(image_array)

recognizer.train(x_train, np.array(y_labels))
recognizer.save(image_dir + '\\trainer-faces.yml')

for label in y_labels:
    label_ids[label] = y_labels.index(label)

with open(image_dir + '\\labels-faces.pickle', "wb") as f:
    pickle.dump(label_ids, f)
