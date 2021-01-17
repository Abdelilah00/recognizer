import os

import cv2
import face_recognition

name = 'Anne'
path = '../datasets/face/' + name + '/'
os.mkdir(path)

cap = cv2.VideoCapture('../src/' + name + '.mp4')

scale = 0.20
scaleV = 5
i = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break

    frame_scale = cv2.resize(frame, (0, 0), None, scale, scale)
    frame_scale = cv2.cvtColor(frame_scale, cv2.COLOR_RGB2GRAY)
    faces = face_recognition.face_locations(frame_scale)

    for face in faces:
        i += 1
        y1, x2, y2, x1 = face
        y1, x2, y2, x1 = y1 * scaleV, x2 * scaleV, y2 * scaleV, x1 * scaleV
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        filename = path + str(i) + '.jpg'
        gray_face = cv2.cvtColor(frame[y1 + 10:y2 - 10, x1 + 10:x2 - 10], cv2.COLOR_RGB2GRAY)
        cv2.imwrite(filename, gray_face)

    # cv2.imshow('gray frame', frame_scale)
    cv2.imshow('rgb frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
