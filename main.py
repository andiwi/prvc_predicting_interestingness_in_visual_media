import os

import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict

import numpy as np
import matplotlib.pyplot as plt

import preprocessing
from feature_extraction import feature_calculation#, obj_recognition
from feature_extraction.feature_processing import scale_features, concat_features, reshape_arrays_1D_to_2D, \
    gen_final_feature_matrix, get_target_vec
from Features import Features
from file_handler import feature_files
from helper import box_filter


def main():
    # the features which should be used.
    feature_names = [
        Features.Symmetry
    ]

    do_preprocessing = False
    calc_features = True

    directory_root = 'D:\\PR aus Visual Computing\\Interestingness17data\\allvideos\\images'
    #directory_root = 'C:\Users\Andreas\Desktop\\testimgs'
    directory_root = '/home/andreas/Desktop/testimgs'
    dir_training_data = os.path.join(directory_root, 'trainingData')
    dir_test_data = os.path.join(directory_root, 'testData')


    # preprocessing
    if (do_preprocessing):
        preprocessing.preprocessing(directory_root)
        print 'preprocessing finished.'

    # calculate features
    if calc_features:
        features = feature_calculation.calc_features(dir_training_data, feature_names)

        # save unscaled features
        feature_files.save_features(dir_training_data, features)
        print 'feature calculation finished.'

    else:
        # load features from files
        features = feature_files.load_features(dir_training_data, feature_names)

    # scale features (because svm is not scale invariant)
    features = scale_features(features)

    # concatenate features
    features = concat_features(features)

    # reshape 1D arrays to 2D arrays
    features = reshape_arrays_1D_to_2D(features)

    # generate final feature matrix
    X = gen_final_feature_matrix(features)

    # TODO do pca analysis

    # get interestingness
    y = get_target_vec(dir_training_data)

    #
    # train and test svm
    #
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C, probability=True)  # Accuracy: 0.87 (+/- 0.07)
    # svc = svm.SVC(kernel='linear', C=C, class_weight={1:10}) #Accuracy: 0.77 (+/- 0.13)
    scores = cross_val_score(svc, X, y, cv=10, scoring='average_precision')
    proba = cross_val_predict(svc, X, y, cv=10, method='predict')
    preditctions = cross_val_predict(svc, X, y, cv=10, method='predict_proba')

    probability = preditctions[:, 0]
    x_vals = np.asarray(range(0,len(probability)))

    #sort probability
    probability = np.sort(probability)

    #smooth curve with averaging window
    proba_smooth = box_filter.smooth(probability, 3)

    #calc second order derivative
    diff_2nd = np.diff(proba_smooth, 2)

    #find first point above threshold
    thresh = 0.001
    limitIdx = None #This position corresponds to the limit between non interesting and interesting shots/key-frames.
    for i in range(len(diff_2nd)):
        if diff_2nd[i] > thresh:
            limitIdx = i
            break

    #DEBUG
    plt.plot(x_vals, probability, 'o')
    plt.plot(x_vals, proba_smooth, 'r-', lw=2)
    plt.plot(x_vals[i], proba_smooth[i], 'go')
    plt.show()
    #DEBUG END

    y_pred = probability < proba_smooth[limitIdx]

    #calc new interestingness probability according to limit point
    proba_unint = probability[np.invert(y_pred)]
    proba_int = probability[y_pred]

    #normalize uninteresting in range 0.5 - 1
    proba_unint_scaled = sklearn.preprocessing.minmax_scale(proba_unint, feature_range=(0.5, 1))
    #normalize interesting in range 0 - 0.5
    proba_int_scaled = sklearn.preprocessing.minmax_scale(proba_int, feature_range=(0, 0.5))

    #set interesting and non interesting according to limitPoint








    #TEST
    #x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    #y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    #z = np.polyfit(x, y, 3)
    #p = np.poly1d(z)
    #p_der = np.poly1d(np.polyder(p,1))
    #p_der2 = np.poly1d(np.polyder(p, 2))
    #xp = np.linspace(-2, 6, 100)
    #_ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p_der2(xp), '--')
    ## plt.ylim(0,1)
    #plt.show()




    #find border between interesting and uninteresting

    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("finished.")


if __name__ == '__main__': main()
