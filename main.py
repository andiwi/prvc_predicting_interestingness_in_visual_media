import os

import random

import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict

import numpy as np
import matplotlib.pyplot as plt

import preprocessing.preprocessing as prvc_preprocessing
from feature_extraction import feature_calculation
from feature_extraction.feature_processing import scale_features, \
    gen_final_feature_matrix, get_target_vec, make_face_bb_equal_col_size, \
    scale_feature_vec
from Features import Features
from file_handler import feature_files
from file_handler.read_gt_file import read_img_dirs_and_gt
from file_handler.read_imgs import read_img_dirs
from helper import box_filter


def main():
    # the features which should be used.
    feature_names = [
        Features.Face_count,
        #Features.Rot_distance,
        Features.Face_bb,
        #Features.Face_bb_full_img,
        #Features.Face_bb_quarter_imgs,
        #Features.Face_bb_eighth_imgs,
        #Features.Tilted_edges,
        #Features.Edge_hist_v0,
        #Features.Edge_hist_v1,
        #Features.Edge_hist_v2,
        #Features.Symmetry,
        #Features.Hsv_hist,
        #Features.DenseSIFT_L0,
        #Features.DenseSIFT_L1,
        #Features.DenseSIFT_L2,
        #Features.Hog_L0,
        #Features.Hog_L1,
        #Features.Hog_L2,
        #Features.Lbp_L0,
        #Features.Lbp_L1,
        #Features.Lbp_L2,
        #Features.Gist,
        #Features.CNN_fc7,
        #Features.CNN_prob
    ]

    do_preprocessing = False
    calc_features = False

    global dir_root
    dir_root = '/home/andreas/Desktop/InterestingnessData16_small'
    #root directories for training and test data
    dir_training_data = os.path.join(dir_root, 'devset')
    dir_test_data = os.path.join(dir_root, 'testset')

    #dicts containing path to images as keys and ground truth as values
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
        #features_test = feature_calculation.calc_features(img_dirs_test, feature_names)
        print 'feature calculation finished.'

    else:
        # load features from files
        features_train = feature_files.load_features(img_dirs_training.keys(), feature_names)
        #features_test = feature_files.load_features(img_dirs_test, feature_names)

    # scale features (because svm is not scale invariant)
    features_train = scale_features(features_train)
    #features_test = scale_features(features_test)

    if Features.Face_bb in feature_names:
       # bring bounding box feature matrices to same shape
       # find matrix with maximal columns and reshape other matrix before concatenating them
       features_train = make_face_bb_equal_col_size(features_train)
       #features_test = make_face_bb_equal_col_size(features_test)

    # generate final feature matrix
    X_train = gen_final_feature_matrix(features_train)
    #X_test = gen_final_feature_matrix(features_test)

    #scale again
    X_train = scale_feature_vec(X_train)
    #X_test = scale_feature_vec(X_test)

    # get interestingness
    y_train = get_target_vec(img_dirs_training)
    #y_test = get_target_vec(img_dirs_test)

    #
    # train and test svm
    #
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='rbf', C=C, probability=True)

    #train svm
    svc.fit(X_train, y_train)
    #classify test set
    y_proba = svc.predict_proba(X_test)

    #TODO for final submission
    ##choose random 10 imgs as test set
    #testIdxs = random.sample(range(0, len(y)), k=10)
    #img_names = read_img_names(dir_training_data)
#
    #img_names_interesting = read_img_names(os.path.join(dir_training_data, 'interesting'))
    #img_names_uninteresting = read_img_names(os.path.join(dir_training_data, 'uninteresting'))
    #img_names = np.concatenate((img_names_interesting, img_names_uninteresting), axis=0)
    #img_names_test = img_names[testIdxs]
#
    #X_train = np.delete(X, testIdxs, axis=0)
    #y_train = np.delete(y, testIdxs, axis=0)
    #X_test = X[testIdxs]
    #y_test = y[testIdxs]
    #

#
    #submission_format = postprocessing.gen_submission_format(img_names_test, y_proba)
    #file_handler.save_submission(submission_format)






    scores = cross_val_score(svc, X, y, cv=10, scoring='average_precision')
    predictions = cross_val_predict(svc, X, y, cv=10, method='predict_proba')

    # classic version
    avg_precision_score_v1 = sklearn.metrics.average_precision_score(y, predictions[:, 1])

    probability = predictions[:, 1]
    x_vals = np.asarray(range(0, len(probability)))

    #sort probability
    probability = np.sort(probability)

    #smooth curve with averaging window
    proba_smooth = box_filter.smooth(probability, 3)

    #calc second order derivative
    diff_2nd = np.diff(proba_smooth, 2)

    #find first point above threshold
    thresh = 0.01
    limitIdx = None #This position corresponds to the limit between non interesting and interesting shots/key-frames.
    for i in range(len(diff_2nd)):
        if diff_2nd[i] > thresh:
            limitIdx = i
            break

    if limitIdx is None:
        limitIdx = i

    #DEBUG
    plt.plot(x_vals, probability, 'o')
    plt.plot(x_vals, proba_smooth, 'r-', lw=2)
    plt.plot(x_vals[limitIdx], proba_smooth[limitIdx], 'go')
    plt.show()
    #DEBUG END

    y_pred = probability > proba_smooth[limitIdx]

    #calc new interestingness probability according to limit point
    proba_unint = probability[np.invert(y_pred)]
    proba_int = probability[y_pred]

    #normalize uninteresting in range 0 - 0.5
    proba_unint_scaled = sklearn.preprocessing.minmax_scale(proba_unint, feature_range=(0, 0.5))
    #normalize interesting in range 0.5-1
    proba_int_scaled = sklearn.preprocessing.minmax_scale(proba_int, feature_range=(0.5, 1))

    #set interesting and non interesting according to limitPoint
    y_proba = np.concatenate((proba_int_scaled, proba_unint_scaled), axis=0)
    avg_precision_score_v2 = sklearn.metrics.average_precision_score(y, y_proba)



    #DEBUG
    #y = [1,1,0,0]
    #y_proba = [0.9, 0.8, 0.4, 0.2]
    #avg_precision_score = sklearn.metrics.average_precision_score(y, y_proba)
    #DEBUG END

    #find border between interesting and uninteresting

    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Mean Average Precision - Version 1: %0.2f" % avg_precision_score_v1)
    print("Mean Average Precision - Version 2: %0.2f" % avg_precision_score_v2)
    print("finished.")


if __name__ == '__main__': main()
