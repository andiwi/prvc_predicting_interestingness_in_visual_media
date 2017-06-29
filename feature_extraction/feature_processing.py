import numpy as np
import os
from sklearn import preprocessing
from Features import Features
from helper import np_helper
from file_handler.read_imgs import read_img_names


def scale_features(features):
    """
    scales all features in features
    :param features:
    :return: scaled_features
    """
    scaled_features = {}
    for name in features:
        interesting = preprocessing.scale(features[name][0])
        uninteresting = preprocessing.scale(features[name][1])
        scaled_features[name] = [interesting, uninteresting]

    return scaled_features


def concat_features(features):
    """
    concatenates interesting and uninteresting features
    :param features:
    :return: concatenated features
    """
    concated_features = {}
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
    :return:
    """
    reshaped_features = features
    for name in features:
        if features[name].ndim == 1:
            reshaped_features[name] = np.reshape(features[name], (len(features[name]), 1))

    return reshaped_features


def gen_final_feature_matrix(features):
    """
    generates the final feature matrix which can be used to train SVM
    :param features:
    :return:
    """
    for name in features:
        try:
            final_feature_mat = np.c_[final_feature_mat, features[name]]
        except NameError:
            # only the first feature will raise this exception
            final_feature_mat = features[name]

    return final_feature_mat


def get_target_vec(dir):
    """
    calculates the target vector. 1 = interesting, 0 = uninteresting
    :param dir:
    :return:
    """
    img_names_interesting = read_img_names(os.path.join(dir, 'interesting'))
    img_names_uninteresting = read_img_names(os.path.join(dir, 'uninteresting'))
    target_interesting = np.ones((len(img_names_interesting),))
    target_uninteresting = np.zeros((len(img_names_uninteresting),))

    y = np.concatenate((target_interesting, target_uninteresting), axis=0)
    return y
