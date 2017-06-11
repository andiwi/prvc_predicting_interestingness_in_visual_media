from sklearn import svm
import numpy as np
import np_helper
import cv2
import os
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from face_detection import face_detection
from crop_imgs import crop_black_borders
from selectTrainingAndTestData import selectTrainingAndTestData
from read_imgs import read_img_names
from feature_calculation import calc_features as feature_calculation, face_count_calculator, img_tilted_calculator


def main():
    do_preprocessing = False
    calc_features = True
    load_features = True

    # which features should be used
    use_face_count = False
    use_rot_distance = False
    use_face_bb = False
    use_tilted_edges = True

    #directory_root = 'D:\\PR aus Visual Computing\\Interestingness17data\\allvideos\\images'
    directory_root = 'C:\Users\Andreas\Desktop\edge histogram problem'
    dir_training_data = os.path.join(directory_root, 'trainingData')
    dir_test_data = os.path.join(directory_root, 'testData')

    #
    # preprocessing
    #
    if (do_preprocessing):
        # filecopy()
        selectTrainingAndTestData(os.path.join(directory_root, 'interesting'),
                                  os.path.join(directory_root, 'uninteresting'), dir_training_data, dir_test_data,
                                  interesting_training_samples=50, interesting_test_samples=50,
                                  uninteresting_training_samples=500, uninteresting_test_samples=500)
        crop_black_borders(os.path.join(dir_training_data, 'interesting'))
        crop_black_borders(os.path.join(dir_training_data, 'uninteresting'))
        crop_black_borders(os.path.join(dir_test_data, 'interesting'))
        crop_black_borders(os.path.join(dir_test_data, 'uninteresting'))

        print 'preprocessing finished.'
    #
    # calculate features
    #
    if (calc_features):

        if (use_face_count or use_rot_distance or use_face_bb):
            directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
            face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
            face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')
            # face_count_interesting = feature_calculation(os.path.join(dir_training_data, 'interesting'), face_count_calculator, (face_frontal_cascade, face_profile_cascade))
            # calc face features
            face_count_interesting, rot_distance_interesting, face_bb_interesting = face_detection(
                os.path.join(dir_training_data, 'interesting'))
            face_count_uninteresting, rot_distance_uninteresting, face_bb_uninteresting = face_detection(
                os.path.join(dir_training_data, 'uninteresting'))

        if (use_tilted_edges):
            # calc camera angle feature (edge orientation histogram)
            tilted_edges_interesting = feature_calculation(os.path.join(dir_training_data, 'interesting'),
                                                           img_tilted_calculator)
            tilted_edges_uninteresting = feature_calculation(os.path.join(dir_training_data, 'uninteresting'),
                                                             img_tilted_calculator)

        #
        # save unscaled features
        #
        if (use_face_count or use_rot_distance or use_face_bb):
            np.savetxt(os.path.join(dir_training_data, 'face_count_interesting.gz'), face_count_interesting)
            np.savetxt(os.path.join(dir_training_data, 'rot_distance_interesting.gz'), rot_distance_interesting)
            np.savetxt(os.path.join(dir_training_data, 'face_bb_interesting.gz'), face_bb_interesting)

            np.savetxt(os.path.join(dir_training_data, 'face_count_uninteresting.gz'), face_count_uninteresting)
            np.savetxt(os.path.join(dir_training_data, 'rot_distance_uninteresting.gz'), rot_distance_uninteresting)
            np.savetxt(os.path.join(dir_training_data, 'face_bb_uninteresting.gz'), face_bb_uninteresting)

        if (use_tilted_edges):
            np.savetxt(os.path.join(dir_training_data, 'tilted_edges_interesting.gz'), tilted_edges_interesting)
            np.savetxt(os.path.join(dir_training_data, 'tilted_edges_uninteresting.gz'), tilted_edges_uninteresting)

        print 'feature calculation finished.'

    if (load_features):
        #
        # load features from files
        #
        if (use_face_count):
            face_count_interesting = np.loadtxt(os.path.join((dir_training_data), 'face_count_interesting.gz'))
            face_count_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'face_count_uninteresting.gz'))

        if (use_rot_distance):
            rot_distance_interesting = np.loadtxt(os.path.join((dir_training_data), 'rot_distance_interesting.gz'))
            rot_distance_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'rot_distance_uninteresting.gz'))

        if (use_face_bb):
            face_bb_interesting = np.loadtxt(os.path.join((dir_training_data), 'face_bb_interesting.gz'))
            face_bb_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'face_bb_uninteresting.gz'))

        if (use_tilted_edges):
            tilted_edges_interesting = np.loadtxt(os.path.join(dir_training_data, 'tilted_edges_interesting.gz'))
            tilted_edges_uninteresting = np.loadtxt(os.path.join(dir_training_data, 'tilted_edges_uninteresting.gz'))

    #
    # scale features (because svm is not scale invariant)
    #
    if (use_face_count):
        face_count_interesting = preprocessing.scale(face_count_interesting)
        face_count_uninteresting = preprocessing.scale(face_count_uninteresting)

    if (use_rot_distance):
        rot_distance_interesting = preprocessing.scale(rot_distance_interesting)
        rot_distance_uninteresting = preprocessing.scale(rot_distance_uninteresting)

    if (use_face_bb):
        face_bb_interesting = preprocessing.scale(face_bb_interesting)
        face_bb_uninteresting = preprocessing.scale(face_bb_uninteresting)

    if (use_tilted_edges):
        tilted_edges_interesting = preprocessing.scale(tilted_edges_interesting)
        tilted_edges_uninteresting = preprocessing.scale(tilted_edges_uninteresting)

    #
    # concatenate features
    #
    if (use_face_count):
        face_count = np.concatenate((face_count_interesting, face_count_uninteresting), axis=0)

    if (use_rot_distance):
        rot_distance = np.concatenate((rot_distance_interesting, rot_distance_uninteresting), axis=0)

    if (use_face_bb):
        # bring bounding box feature matrices to same shape
        # find matrix with maximal columns and reshape other matrixe before concatenating them
        face_bb_interesting, face_bb_uninteresting = np_helper.numpy_fillcolswithzeros(face_bb_interesting,
                                                                                       face_bb_uninteresting)
        face_bb = np.concatenate((face_bb_interesting, face_bb_uninteresting), axis=0)

    if (use_tilted_edges):
        tilted_edges = np.concatenate((tilted_edges_interesting, tilted_edges_uninteresting), axis=0)

    #
    # reshape 1D arrays to 2D arrays
    #
    if (use_face_count):
        face_count = np.reshape(face_count, (len(face_count), 1))

    if (use_rot_distance):
        rot_distance = np.reshape(rot_distance, (len(rot_distance), 1))

    if (use_tilted_edges):
        tilted_edges = np.reshape(tilted_edges, (len(tilted_edges), 1))

    #
    # generate final feature matrix
    #
    if (use_face_count):
        try:
            X = np.c_[X, face_count]
        except NameError:
            X = face_count

    if (use_rot_distance):
        try:
            X = np.c_[X, rot_distance]
        except NameError:
            X = rot_distance

    if (use_face_bb):
        try:
            X = np.c_[X, face_bb]
        except NameError:
            X = face_bb
    if (use_tilted_edges):
        try:
            X = np.c_[X, tilted_edges]
        except NameError:
            X = tilted_edges

    #
    # get interestingness
    #
    img_names_interesting = read_img_names(os.path.join(dir_training_data, 'interesting'))
    img_names_uninteresting = read_img_names(os.path.join(dir_training_data, 'uninteresting'))
    target_interesting = np.ones((len(img_names_interesting),))
    target_uninteresting = np.zeros((len(img_names_uninteresting),))

    y = np.concatenate((target_interesting, target_uninteresting), axis=0)

    #
    # train and test svm
    #
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C)  # Accuracy: 0.87 (+/- 0.07)
    # svc = svm.SVC(kernel='linear', C=C, class_weight={1:10}) #Accuracy: 0.77 (+/- 0.13)
    scores = cross_val_score(svc, X, y, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("finished.")

    '''
    use_face_count = True
    use_rot_distance = True
    use_face_bb = False
    
    face_count: Mean Average Precision: 0.08 (+/- 0.04)
    rot: Mean Average Precision: 0.22 (+/- 0.28)
    face_bb: Mean Average Precision: 0.21 (+/- 0.61)
    
    face_count & rot: Mean Average Precision: 0.16 (+/- 0.17)
    face_count & face_bb: Mean Average Precision: 0.21 (+/- 0.61)
    rot & face_bb: Mean Average Precision: 0.21 (+/- 0.61)
    
    face_count & rot & face_bb: Mean Average Precision: 0.21 (+/- 0.61)
    '''


if __name__ == '__main__': main()
