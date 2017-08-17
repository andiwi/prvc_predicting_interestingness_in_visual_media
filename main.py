import os

import random
from collections import OrderedDict
from warnings import warn

import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import file_handler
from postprocessing.postprocessing import gen_submission_format
import preprocessing.preprocessing as prvc_preprocessing
from feature_extraction import feature_calculation
from feature_extraction.feature_processing import scale_features, \
    gen_final_feature_matrix, get_target_vec, make_face_bb_equal_col_size, make_face_bb_train_test_equal_col_size, \
    gen_feature_matrices_per_feature
from Features import Features
from file_handler import feature_files, save_submission
from file_handler.read_gt_file import read_img_dirs_and_gt
from file_handler.read_imgs import read_img_dirs
from helper import box_filter
from predict import predict


def main():
    # the features which should be used.
    feature_names = [
        # Features.Face_count,
        # Features.Rot_distance,
        # Features.Face_bb,
        # Features.Face_bb_full_img,
        # Features.Face_bb_quarter_imgs,
        # Features.Face_bb_eighth_imgs,
        # Features.Tilted_edges,
        # Features.Edge_hist_v0,
        # Features.Edge_hist_v1,
        # Features.Edge_hist_v2,
        # Features.Symmetry,
        # Features.Hsv_hist,
         Features.DenseSIFT_L0,
        # Features.DenseSIFT_L1,
        # Features.DenseSIFT_L2,
        # Features.Hog_L0,
        # Features.Hog_L1,
        # Features.Hog_L2,
        # Features.Lbp_L0,
        # Features.Lbp_L1,
        # Features.Lbp_L2,
         Features.Gist,
        # Features.CNN_fc7,
        # Features.CNN_prob
    ]

    runname = 1
    do_preprocessing = False  # use this only at your first run on the dataset
    calc_features = False  # calculates the selected features
    use_second_dev_classification_method = False # True: classifies with second order deviation method
    global dir_root # the root directory of your data
    dir_root = 'C:\Users\Andreas\Desktop\prvc\InterestingnessData2016'

#######################
###STOP EDITING HERE###
#######################

    # root directories for training and test data
    dir_training_data = os.path.join(dir_root, 'devset')
    dir_test_data = os.path.join(dir_root, 'testset')

    # dicts containing path to images as keys and ground truth as values
    img_dirs_training = read_img_dirs_and_gt(dir_training_data)
    img_dirs_test = read_img_dirs(dir_test_data)

    # preprocessing
    if do_preprocessing:
        prvc_preprocessing.preprocessing(img_dirs_training.keys())
        prvc_preprocessing.preprocessing(img_dirs_test)
        print 'preprocessing finished.'

    # calculate features
    if calc_features:
        features_train = feature_calculation.calc_features(img_dirs_training.keys(), feature_names)
        features_test = feature_calculation.calc_features(img_dirs_test, feature_names)
        print 'feature calculation finished.'

    else:
        # load features from file
        features_train = feature_files.load_features(img_dirs_training.keys(), feature_names)
        features_test = feature_files.load_features(img_dirs_test, feature_names)

    print('features loaded.')

    if Features.Face_bb in feature_names:
        # bring bounding box feature matrices to same shape
        # find matrix with maximal columns and reshape other matrix before concatenating them
        features_train = make_face_bb_equal_col_size(features_train)
        features_test = make_face_bb_equal_col_size(features_test)
        features_train, features_test = make_face_bb_train_test_equal_col_size(features_train, features_test)

    X_trains = gen_feature_matrices_per_feature(features_train)
    X_tests = gen_feature_matrices_per_feature(features_test)

    # scale features (because svm is not scale invariant)
    X_trains_scaled = scale_features(X_trains)
    X_tests_scaled = scale_features(X_tests)

    # generate final feature matrix
    X_train = gen_final_feature_matrix(X_trains)
    X_test = gen_final_feature_matrix(X_tests)

    X_train_scaled = gen_final_feature_matrix(X_trains_scaled)
    X_test_scaled = gen_final_feature_matrix(X_tests_scaled)

    #DEBUG save
    #np.savetxt('C:\Users\Andreas\Desktop\\X_train_fc7.txt.gz', X_train)
    #np.savetxt('C:\Users\Andreas\Desktop\\X_train_fc7.txt.gz_scaled.txt.gz', X_train_scaled)
    #np.savetxt('C:\Users\Andreas\Desktop\\X_test_fc7.txt.gz', X_test)
    #np.savetxt('C:\Users\Andreas\Desktop\\X_test_fc7.txt.gz_scaled.txt.gz', X_test_scaled)

    # get interestingness
    y_train = get_target_vec(img_dirs_training)


    #upsampling of class 'interesting' via SMOTE
    #sm = SMOTE()
    #X_train_upsampled, y_train_upsampled = sm.fit_sample(X_train, y_train)
    #X_train = X_train_upsampled
    #y_train = y_train_upsampled

    
    #
    # train and test svm
    #
    #C = 0.125  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced') #class_weight='balanced'
    #results_1 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_1 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 0.25  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_2 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_2 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 0.5  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_3 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_3 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 1  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_4 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_4 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 2  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_5 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_5 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 4  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_6 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_6 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 8  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_7 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_7 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)
#
    #C = 16  # SVM regularization parameter
    #svc = svm.SVC(kernel='rbf', C=C, class_weight='balanced')  # class_weight='balanced'
    #results_8 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled_8 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
    #                         use_second_dev_classification_method)


    # submission_format = gen_submission_format(results_1)
    # save_submission.save_submission(submission_format, 1)
    # submission_format = gen_submission_format(results_scaled_1)
    # save_submission.save_submission(submission_format, 2)
    #
    # submission_format = gen_submission_format(results_2)
    # save_submission.save_submission(submission_format, 3)
    # submission_format = gen_submission_format(results_scaled_2)
    # save_submission.save_submission(submission_format, 4)
    #
    # submission_format = gen_submission_format(results_3)
    # save_submission.save_submission(submission_format, 5)
    # submission_format = gen_submission_format(results_scaled_3)
    # save_submission.save_submission(submission_format, 6)
    #
    # submission_format = gen_submission_format(results_4)
    # save_submission.save_submission(submission_format, 7)
    # submission_format = gen_submission_format(results_scaled_4)
    # save_submission.save_submission(submission_format, 8)
    #
    # submission_format = gen_submission_format(results_5)
    # save_submission.save_submission(submission_format, 9)
    # submission_format = gen_submission_format(results_scaled_5)
    # save_submission.save_submission(submission_format, 10)
    #
    # submission_format = gen_submission_format(results_6)
    # save_submission.save_submission(submission_format, 11)
    # submission_format = gen_submission_format(results_scaled_6)
    # save_submission.save_submission(submission_format, 12)
    #
    # submission_format = gen_submission_format(results_7)
    # save_submission.save_submission(submission_format, 13)
    # submission_format = gen_submission_format(results_scaled_7)
    # save_submission.save_submission(submission_format, 14)
    #
    # submission_format = gen_submission_format(results_8)
    # save_submission.save_submission(submission_format, 15)
    # submission_format = gen_submission_format(results_scaled_8)
    # save_submission.save_submission(submission_format, 16)

    #LAPI Settings for HSVHist + GIST ---MAP should be 0.1714
    #print("svm.SVC(kernel='poly', degree=18, gamma=2, class_weight={1 : 10})")
    #svc = svm.SVC(kernel='poly', degree=18, gamma=2, class_weight={1 : 10})
    #results = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    #results_scaled = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test, use_second_dev_classification_method)
    
    #svc = svm.SVC(kernel='poly', degree=18, gamma=2)

    #LAPI Settings for DSIFT + GIST ---MAP should be 0.1398
    print("svm.SVC(kernel='poly', degree=3, gamma=32, class_weight={1: 10})")
    svc = svm.SVC(kernel='poly', degree=3, gamma=32, class_weight={1: 10})
    #svc = svm.SVC(kernel='poly', degree=3, gamma=32)
    results = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    results_scaled = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test, use_second_dev_classification_method)

    print("svm.SVC(kernel='poly', degree=3, gamma=32, class_weight='balanced')")
    svc = svm.SVC(kernel='poly', degree=3, gamma=32, class_weight='balanced')
    results_2 = predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method)
    results_scaled_2 = predict(svc, X_train_scaled, y_train, X_test_scaled, features_test,
                             use_second_dev_classification_method)

    print("save results")
    submission_format = gen_submission_format(results)
    save_submission.save_submission(submission_format, 1)
    submission_format = gen_submission_format(results_scaled)
    save_submission.save_submission(submission_format, 2)

    submission_format = gen_submission_format(results_2)
    save_submission.save_submission(submission_format, 3)
    submission_format = gen_submission_format(results_scaled_2)
    save_submission.save_submission(submission_format, 4)



    '''
    #read ground truth of testset
    img_dirs_test = read_img_dirs_and_gt(dir_test_data)
    y_test = get_target_vec(img_dirs_test)


    
    print('UNSCALED')
    print('LAPI 1:10')
    svc = svm.SVC(kernel='poly', degree=18, gamma=2, class_weight={1: 10})
    scores = cross_val_score(svc, X_test, y_test, cv=3, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('LAPI')
    svc = svm.SVC(kernel='poly', degree=18, gamma=2)
    scores = cross_val_score(svc, X_test, y_test, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('LAPI c=0.1')
    svc = svm.SVC(kernel='poly', degree=18, gamma=2, C=0.1)
    scores = cross_val_score(svc, X_test, y_test, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('LAPI balanced')
    svc = svm.SVC(kernel='poly', degree=18, gamma=2, class_weight='balanced')
    scores = cross_val_score(svc, X_test, y_test, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('LAPI like libsvm balanced')
    svc = svm.SVC(kernel='poly', degree=18, gamma=2, class_weight='balanced', cache_size=100)
    scores = cross_val_score(svc, X_test, y_test, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('LAPI like libsvm 1 to 10')
    svc = svm.SVC(kernel='poly', degree=18, gamma=2, class_weight={1: 10}, cache_size=100)
    scores = cross_val_score(svc, X_test, y_test, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('LAPI like libsvm 1 to 10, C=0.25')
    svc = svm.SVC(kernel='poly', C=0.25, degree=18, gamma=2, class_weight={1: 10}, cache_size=100)
    scores = cross_val_score(svc, X_test, y_test, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    '''
    print("finished.")

if __name__ == '__main__': main()
