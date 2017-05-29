from draw_rule_of_third_lines import draw_rule_of_thirds_lines
from filecopy import filecopy
from draw_phi_grid import draw_phi_grid
from calc_histogram import calc_histograms, calc_histograms_plus_img, calc_histograms_normalized, calc_histograms_bw
from face_detection import face_detection, face_to_img_ratios_to_csv
from crop_imgs import crop_black_borders
from selectTrainingAndTestData import selectTrainingAndTestData
from read_imgs import read_img_names
from sklearn import svm, datasets
import numpy as np
import np_helper
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

def main():
    do_preprocessing = False
    calc_features = False
    load_features = True

    #which features should be used
    use_face_count = True
    use_rot_distance = True
    use_face_bb = True


    directory_original = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting'
    directory_cropped = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped'

    directory_root = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images'
    dir_training_data = os.path.join(directory_root, 'trainingData')
    dir_test_data = os.path.join(directory_root, 'testData')

    #
    #preprocessing
    #
    if(do_preprocessing):
        selectTrainingAndTestData(os.path.join(directory_root, 'interesting'), os.path.join(directory_root, 'uninteresting'), dir_training_data, dir_test_data, interesting_training_samples=50, interesting_test_samples=50, uninteresting_training_samples=500, uninteresting_test_samples=500)
        crop_black_borders(os.path.join(dir_training_data, 'interesting'))
        crop_black_borders(os.path.join(dir_training_data, 'uninteresting'))
        crop_black_borders(os.path.join(dir_test_data, 'interesting'))
        crop_black_borders(os.path.join(dir_test_data, 'uninteresting'))

    #
    #calculate features
    #
    if(calc_features):
        #calc face features
        face_count_interesting, rot_distance_interesting, face_bb_interesting = face_detection(os.path.join(dir_training_data, 'interesting'))
        face_count_uninteresting, rot_distance_uninteresting, face_bb_uninteresting = face_detection(os.path.join(dir_training_data, 'uninteresting'))

        #
        # save unscaled features
        #
        np.savetxt(os.path.join(dir_training_data, 'face_count_interesting.gz'), face_count_interesting)
        np.savetxt(os.path.join(dir_training_data, 'rot_distance_interesting.gz'), rot_distance_interesting)
        np.savetxt(os.path.join(dir_training_data, 'face_bb_interesting.gz'), face_bb_interesting)

        np.savetxt(os.path.join(dir_training_data, 'face_count_uninteresting.gz'), face_count_uninteresting)
        np.savetxt(os.path.join(dir_training_data, 'rot_distance_uninteresting.gz'), rot_distance_uninteresting)
        np.savetxt(os.path.join(dir_training_data, 'face_bb_uninteresting.gz'), face_bb_uninteresting)

    if(load_features):
        #
        # load features from files
        #
        face_count_interesting = np.loadtxt(os.path.join((dir_training_data), 'face_count_interesting.gz'))
        rot_distance_interesting = np.loadtxt(os.path.join((dir_training_data), 'rot_distance_interesting.gz'))
        face_bb_interesting = np.loadtxt(os.path.join((dir_training_data), 'face_bb_interesting.gz'))

        face_count_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'face_count_uninteresting.gz'))
        rot_distance_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'rot_distance_uninteresting.gz'))
        face_bb_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'face_bb_uninteresting.gz'))

    #
    # scale features (because svm is not scale invariant)
    #
    face_count_interesting = preprocessing.scale(face_count_interesting)
    rot_distance_interesting = preprocessing.scale(rot_distance_interesting)
    face_bb_interesting = preprocessing.scale(face_bb_interesting)

    face_count_uninteresting = preprocessing.scale(face_count_uninteresting)
    rot_distance_uninteresting = preprocessing.scale(rot_distance_uninteresting)
    face_bb_uninteresting = preprocessing.scale(face_bb_uninteresting)


    #
    # concatenate face bb features
    #

    face_count = np.concatenate((face_count_interesting, face_count_uninteresting), axis=0)
    rot_distance = np.concatenate((rot_distance_interesting, rot_distance_uninteresting), axis=0)

    # bring bounding box feature matrices to same shape
    # find matrix with maximal columns and reshape other matrixe before concatenating them
    face_bb_interesting, face_bb_uninteresting = np_helper.numpy_fillcolswithzeros(face_bb_interesting, face_bb_uninteresting)
    face_bb = np.concatenate((face_bb_interesting, face_bb_uninteresting), axis=0)


    #
    # generate final feature matrix
    #
    if(use_face_count):
        try:
            X = np.c_[X, face_count]
        except NameError:
            X = face_count

    if(use_rot_distance):
        try:
            X = np.c_[X, rot_distance]
        except NameError:
            X = rot_distance

    if(use_face_bb):
        try:
            X = np.c_[X, face_bb]
        except NameError:
            X = face_bb


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
    svc = svm.SVC(kernel='linear', C=C) #Accuracy: 0.87 (+/- 0.07)
    #svc = svm.SVC(kernel='linear', C=C, class_weight={1:10}) #Accuracy: 0.77 (+/- 0.13)
    scores = cross_val_score(svc, X, y, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("finished.")





#    filecopy()
#    crop_black_borders(directory_original)
#    draw_rule_of_thirds_lines(directory_cropped)
#    draw_phi_grid(directory_cropped)
#    calc_histograms(directory_cropped)
#    calc_histograms_normalized(directory_cropped)
#    calc_histograms_plus_img(directory_cropped)
#    calc_histograms_bw(directory_cropped)

#    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
#    face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
#    face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')
#    #face_detection(directory_cropped, face_frontal_cascade, face_profile_cascade)
#    face_to_img_ratios_to_csv(directory_cropped + '\\faces\\', face_frontal_cascade, face_profile_cascade)

if __name__ == '__main__':main()

