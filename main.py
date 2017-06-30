import os

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score

import preprocessing
from feature_extraction import feature_calculation, obj_recognition
from feature_extraction.feature_processing import scale_features, concat_features, reshape_arrays_1D_to_2D, \
    gen_final_feature_matrix, get_target_vec
from Features import Features
from file_handler import feature_files


def main():
    # the features which should be used.
    feature_names = [
        Features.Face_count
    ]

    do_preprocessing = False
    calc_features = False

    directory_root = 'D:\\PR aus Visual Computing\\Interestingness17data\\allvideos\\images'
    # directory_root = 'C:\Users\Andreas\Desktop\\testimgs'
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
    svc = svm.SVC(kernel='linear', C=C)  # Accuracy: 0.87 (+/- 0.07)
    # svc = svm.SVC(kernel='linear', C=C, class_weight={1:10}) #Accuracy: 0.77 (+/- 0.13)
    scores = cross_val_score(svc, X, y, cv=10, scoring='average_precision')
    print("Mean Average Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("finished.")


if __name__ == '__main__': main()
