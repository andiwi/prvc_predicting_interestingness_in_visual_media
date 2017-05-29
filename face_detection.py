import numpy as np
import cv2
import os
import csv

from np_helper import numpy_fillwithzeros
from Geometry import Rect
from read_imgs import read_img_names, read_img
from rule_of_thirds import distance_to_grid_corner

def face_to_img_ratios_to_csv(directory, face_frontal_cascade = None, face_profile_cascade = None):
    '''
    calculates the ratio between image height and face heights and writes them in a csv file.
    saves csv file at './face_img_ratios.csv'
    :param directory: path to images (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped\\faces')
    :return: 
    '''
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    if face_frontal_cascade is None:
        face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')

    if face_profile_cascade is None:
        face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    imgNames = read_img_names(directory)

    csvfile = open(directory + '\\face_to_img_ratios.csv', 'wb')
    csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    csvwriter.writerow(['shotname', 'faces_detected', 'heights (relative to img height)'])

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        faces_height = calc_face_to_img_height_ratio(img, face_frontal_cascade, face_profile_cascade)
        csvwriter.writerow([imgName, len(faces_height), ",".join(map(str, faces_height.values()))])

    csvfile.close()

def calc_face_to_img_height_ratio(img, face_frontal_cascade = None, face_profile_cascade = None):
    '''
    calculates the ratio between image height and face height
    :param img: 
    :return: dict(face, percentage of face height relative to image height) 
    '''
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    if face_frontal_cascade is None:
        face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')

    if face_profile_cascade is None:
        face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    faces_height = dict()

    rect_faces_final, rect_faces_frontal, rect_faces_profile = detect_faces(img, face_frontal_cascade, face_profile_cascade)

    img_height, img_width, channels = img.shape

    for face in rect_faces_final:
        relative_height = (face.h * 100) / img_height
        faces_height[face] = relative_height

    return faces_height

def detect_faces(img, face_frontal_cascade = None, face_profile_cascade = None):
    '''
    detects faces in image with viola jones algorithm using frontal face and profileface haar features.
    :param img: opencv image
    :param face_frontal_cascade: opencv cascade classifier for frontal faces
    :param face_profile_cascade: opencv cascade classifier for profile faces
    :return: list of Rect containing face bounding boxes, list of Rect containing all frontal face bounding boxes, list of Rect containing all profile face bounding boxes
    '''
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    if face_frontal_cascade is None:
        face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')

    if face_profile_cascade is None:
        face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rect_faces_all = []
    rect_faces_frontal = []
    rect_faces_profile = []
    rect_faces_final = []

    # add all faces to rect_faces_all list
    faces_frontal = face_frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_frontal:
        rect_faces_all.append(Rect(x, y, w, h))
        rect_faces_frontal.append(Rect(x, y, w, h))

    faces_profile = face_profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces_profile:
        rect_faces_all.append(Rect(x, y, w, h))
        rect_faces_profile.append(Rect(x, y, w, h))

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

    return rect_faces_final, rect_faces_frontal, rect_faces_profile

def face_detection(directory, face_frontal_cascade = None, face_profile_cascade = None): #TODO rename method to face_feature_calculation
    '''
    detects faces with viola jones algorithm using frontal face and profileface haar features. 
    NOTE: returns unscaled feature vectors (numpy arrays)
    :param directory: path to images (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: face_count_np              ...number of faces
             rule_of_thirds_distance_np ...distance of biggest face center to nearest rule of thirds grid corner
             imgs_feature_matrix        ...face bounding boxes x,y,w,h, x,y,w,h,...
    '''
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    if face_frontal_cascade is None:
        face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')

    if face_profile_cascade is None:
        face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')


    imgNames = read_img_names(directory)


    face_count = [] #number of faces per img
    rule_of_thirds_distance = [] #smallest euclidean distance between center of biggest face to rule of thirds corner
    imgs_feature_matrix = []

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))

        rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(img, face_frontal_cascade, face_profile_cascade)

        #sort list in descending order
        rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

        #append faces to feature matrix
        #convert rect list to list with entries x,y,w,h entries
        faces = []
        first_rect = True
        for rect in rect_faces:
            faces.insert(len(faces), rect.x)
            faces.insert(len(faces), rect.y)
            faces.insert(len(faces), rect.w)
            faces.insert(len(faces), rect.h)

            if(first_rect):
                first_rect = False
                rule_of_thirds_distance.append(distance_to_grid_corner(img, np.array([rect.x+rect.w/2, rect.y+rect.h/2])))

        if(len(rect_faces) == 0):
            rule_of_thirds_distance.append(0)

        imgs_feature_matrix.append(faces)
        face_count.append(len(rect_faces))

    imgs_feature_matrix_np = np.asarray(imgs_feature_matrix)
    #fill cells in numpy array with zeros
    imgs_feature_matrix_np = numpy_fillwithzeros(imgs_feature_matrix_np)

    face_count_np = np.asarray(face_count)
    rule_of_thirds_distance_np = np.asarray(rule_of_thirds_distance)

    #imgs_feature_matrix_np = np.c_[rule_of_thirds_distance_np, imgs_feature_matrix_np]
    #imgs_feature_matrix_np = np.c_[face_count_np, imgs_feature_matrix_np]

    return face_count_np, rule_of_thirds_distance_np, imgs_feature_matrix_np

