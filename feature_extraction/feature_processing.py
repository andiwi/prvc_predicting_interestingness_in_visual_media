import numpy as np
from sklearn import preprocessing
from Features import Features
from helper import np_helper


def scale_features(features):
    """
    scales all feature vectors in features
    :param features:
    :type features: dict
    :return: scaled features
    :rtype: dict
    """

    scaled_features = {}
    for img_dir in features:
        scaled_features[img_dir] = dict()
        for feature_name in features[img_dir]:
            if Features.is_single_val_feature(feature_name):
                scaled_features[img_dir][feature_name] = features[img_dir][feature_name]
            else:
                scaled_features[img_dir][feature_name] = preprocessing.scale(features[img_dir][feature_name])

    return scaled_features

def scale_feature_vec(feature_vecs):
    """
    scales each feature vector of feature vecs
    :param feature_vecs: the feature vectors
    :type feature_vecs: np.array
    :return: scaled feature vectors
    :rtype: np.array
    """
    scaled_feature_vecs = np.apply_along_axis(preprocessing.scale, 1, feature_vecs)
    return scaled_feature_vecs


def concat_features(features):
    """
    warn: this method is deprecated
    concatenates interesting and uninteresting features
    :param features:
    :return: concatenated features
    """
    concated_features = {}

    for img_dir in features:
        concated_features[img_dir] = dict()
        for feature_name in features[img_dir]:
            if feature_name == Features.Face_bb:
                # bring bounding box feature matrices to same shape
                # find matrix with maximal columns and reshape other matrix before concatenating them
                #TODO
                print('TODO')
            else:
                concated_features[img_dir]

    for name in features:
        if name == Features.Face_bb:
            # bring bounding box feature matrices to same shape
            # find matrix with maximal columns and reshape other matrixe before concatenating them
            interesting, uninteresting = np_helper.numpy_fillcolswithzeros(features[name][0], features[name][1])
            concated_features[name] = np.concatenate((interesting, uninteresting), axis=0)
        else:
            concated_features[name] = np.concatenate((features[name][0], features[name][1]), axis=0)

    return concated_features


def reshape_arrays_1D_to_2D(features):
    """
    reshapes 1D arrays in features to 2D arrays
    :param features:
    :type features: dict
    :return: reshaped features
    :rtype: dict
    """

    reshaped_features = features

    for img_dir in features:
        for feature_name in features[img_dir]:
            if features[img_dir][feature_name].ndim == 1:
                reshaped_features[img_dir][feature_name] = np.reshape(features[img_dir][feature_name], (len(features[img_dir][feature_name]), 1))

    '''
    reshaped_features = features
    for name in features:
        if features[name].ndim == 1:
            reshaped_features[name] = np.reshape(features[name], (len(features[name]), 1))
    '''
    return reshaped_features


def gen_final_feature_matrix(features):
    """
    generates the final feature matrix which can be used to train SVM
    :param features:
    :type features: dict
    :return: final feature matrix
    :rtype: np.array
    """

    final_feature_mat = []

    for img_dir in features:
        final_feature_vec = []
        for feature_name in features[img_dir]:
            if Features.is_single_val_feature(feature_name):
                final_feature_vec.append(np.asscalar(features[img_dir][feature_name]))
            else:
                final_feature_vec.extend(features[img_dir][feature_name].tolist())

        final_feature_mat.append(final_feature_vec)

    return np.asarray(final_feature_mat)


def get_target_vec(img_dirs):
    """
    generates the final target vector which can be used to train SVM
    1 = interesting, 0 = uninteresting
    :param img_dirs:
    :type img_dirs: dict
    :return: the final target vector
    :rtype: np.array
    """

    target_vec = []

    for img_dir in img_dirs:
        target_vec.append(img_dirs[img_dir])

    return np.asarray(target_vec)

def make_face_bb_equal_col_size(features):
    """
    detects feature vector of image with the most columns and adds zeros to the other feature vectors so that all
    feature vectors of face bb feature have same amount of columns
    :param features: the features
    :type features: dict
    :return: dict with equal column size of face bb feature
    :rtype: dict
    """

    #find max face_bb vector
    max_col_size = 0
    for img_dir in features:
        c = features[img_dir][Features.Face_bb].shape[0]
        if c > max_col_size:
            max_col_size = c

    #fill other vectors with zeros
    for img_dir in features:
        c = features[img_dir][Features.Face_bb].shape[0]

        zeros = np.zeros(max_col_size - c)
        features[img_dir][Features.Face_bb] = np.concatenate((features[img_dir][Features.Face_bb], zeros))
    return features
