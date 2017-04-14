import numpy as np
import cv2
import os
import csv
from Geometry import Rect
from read_imgs import read_img_names, read_img

def face_to_img_ratios_to_csv(directory):
    '''
    calculates the ratio between image height and face heights and writes them in a csv file.
    saves csv file at './face_img_ratios.csv'
    :param directory: path to images (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped\\faces')
    :return: 
    '''
    imgNames = read_img_names(directory)

    csvfile = open(directory + '\\face_to_img_ratios.csv', 'wb')
    csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    csvwriter.writerow(['shotname', 'faces_detected', 'heights (relative to img height)'])

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        faces_height = calc_face_to_img_height_ratio(img)
        csvwriter.writerow([imgName, len(faces_height), ",".join(map(str, faces_height.values()))])

    csvfile.close()

def calc_face_to_img_height_ratio(img):
    '''
    calculates the ratio between image height and face height
    :param img: 
    :return: dict(face, percentage of face height relative to image height) 
    '''
    faces_height = dict()

    faces = detect_faces(img)

    img_height, img_width, channels = img.shape

    for face in faces:
        relative_height = (face.h * 100) / img_height
        faces_height[face] = relative_height

    return faces_height

def detect_faces(img):
    '''
    detects faces in image with viola jones algorithm using frontal face and profileface haar features.
    :param img: 
    :return: list of Rect containing face bounding boxes
    '''
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    face_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
    face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rect_faces_all = []
    rect_faces_final = []

    # add all faces to rect_faces_all list
    faces_frontal = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_frontal:
        rect_faces_all.append(Rect(x, y, w, h))

    faces_profile = face_profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces_profile:
        rect_faces_all.append(Rect(x, y, w, h))

    # sort list by area of rectangles
    rect_faces_all.sort(key=lambda rect: rect.area(), reverse=True)

    # add largest rectangle to final list
    # add rectangle if no overlap (or max. overlap of 10% of area) with rectangles of final list
    for rect in rect_faces_all:
        overlap = False
        for rect_2 in rect_faces_final:
            if Rect.overlap(rect, rect_2):
                # calculate intersection area
                # do not add rectangle if max overlap is 10% of area of bigger rectangle
                rect_intersection = Rect.intersection(rect, rect_2)
                if rect_intersection.area() > rect_2.area() * 0.1:
                    overlap = True
        if not overlap:
            rect_faces_final.append(rect)

    return rect_faces_final

def face_detection(directory):
    '''
    detects faces with viola jones algorithm using frontal face and profileface haar features. saves images into ./faces/
    :param directory: path to images (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    face_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
    face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rect_faces_all = []
        rect_faces_final = []

        #add all faces to rect_faces_all list
        faces_frontal = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        for (x, y, w, h) in faces_frontal:
            rect_faces_all.append(Rect(x,y,w,h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        faces_profile = face_profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (x, y, w, h) in faces_profile:
            rect_faces_all.append(Rect(x,y,w,h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #sort list by area of rectangles
        rect_faces_all.sort(key=lambda rect: rect.area(), reverse=True)

        #add largest rectangle to final list
        #add rectangle if no overlap (or max. overlap of 10% of area) with rectangles of final list
        for rect in rect_faces_all:
            overlap = False
            for rect_2 in rect_faces_final:
                if Rect.overlap(rect, rect_2):
                    # calculate intersection area
                    # do not add rectangle if max overlap is 10% of area of bigger rectangle
                    rect_intersection = Rect.intersection(rect, rect_2)
                    if rect_intersection.area() > rect_2.area() * 0.1:
                        overlap = True
            if not overlap:
                rect_faces_final.append(rect)

        #draw all final rects
        for rect in rect_faces_final:
            cv2.rectangle(img, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 0, 255), 2)

        if rect_faces_final:
            # save img
            if not os.path.exists(directory + '\\faces\\'):
                os.makedirs(directory + '\\faces\\')

            imgPath = directory + '\\faces\\' + imgName
            success = cv2.imwrite(imgPath, img)

    print("Finished")