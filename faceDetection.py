import numpy as np
import cv2
import os
from readImgs import readImgNames, readImg

def faceDetection():
    directory = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\faces'
    directory_haarfeatures = 'C:\\opencv\\sources\\data\haarcascades\\'
    face_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
    face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_eye.xml')

    imgNames = readImgNames(directory)

    for imgName in imgNames:
        img = readImg(os.path.join(directory, imgName))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        detectFace = False
        for (x, y, w, h) in faces:
            detectFace = True
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        faces_profile = face_profile_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in faces_profile:
            detectFace = True
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        if detectFace:
            # save img
            if not os.path.exists(directory + '\\detected\\'):
                os.makedirs(directory + '\\detected\\')

            imgPath = directory + '\\detected\\' + imgName
            success = cv2.imwrite(imgPath, img)

    print("Finished")