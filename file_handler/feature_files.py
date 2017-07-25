import os
import platform
from collections import OrderedDict
from warnings import warn

import numpy as np
import gzip
from Features import Features
from feature_extraction.load_precalc_features import load_precalc_feature


def save_features(img_dir, feature_name, feature):
    """
    saves features of image in .txt file (compressed in gzip format)
    :param img_dir: path of the image
    :type img_dir: str
    :param feature_name: name of the feature (should be one of Features enum)
    :type feature_name: str
    :param feature: the feature vector
    :type feature: np.array
    :return:
    :rtype:
    """

    #create feature file path
    feature_file_path = os.path.join(img_dir[:img_dir.find('videos')], 'features', 'Features_From_TUWien', 'Image_Subtask', feature_name, img_dir[img_dir.find('videos')+7:] + '.txt.gz')

    if platform.system() == 'Linux':
        dirs = feature_file_path[:feature_file_path.rfind('/')] #Linux system
    else:
        dirs = feature_file_path[:feature_file_path.rfind('\\')] #windows system

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    if Features.is_single_val_feature(feature_name):
        #feature is a single value
        with gzip.open(feature_file_path, "w") as f:
            f.write(str(feature))
    else:
        np.savetxt(feature_file_path, feature, newline=' ')

def load_features(img_dirs, feature_names):
    """
    loads features from file
    :param img_dirs: a list of image paths
    :type img_dirs: list
    :param feature_names: a list of feature names
    :type feature_names: list
    :return: dict (img_dir, features) features is also a dict (feature_name, feature vector)
    :rtype: dict
    """

    features = OrderedDict()

    for img_dir in img_dirs:
        features[img_dir] = OrderedDict()
        for feature_name in feature_names:
            if Features.is_TU_feature(feature_name):
                feature = _load_TU_feature(img_dir, feature_name)

                if feature is None:
                    #replace with zeros of correct vector size
                    feature = np.zeros(features[img_dirs[0]][feature_name].shape)

            else:
                feature = load_precalc_feature(img_dir, feature_name)

            features[img_dir][feature_name] = feature
    return features

def _load_TU_feature(img_dir, feature_name):
    feature_file_path = os.path.join(img_dir[:img_dir.find('videos')], 'features', 'Features_From_TUWien',
                                     'Image_Subtask', feature_name,
                                     img_dir[img_dir.find('videos') + 7:] + '.txt.gz')

    if os.path.isfile(feature_file_path):
        return np.loadtxt(feature_file_path)
    else:
        warn('File does not exist. You have to calculate the features before loading it. {}'.format(feature_file_path))
        return None

def TU_feature_file_exists(img_dir, feature_name):
    """
    checks if a feature file for image and given feature already exists
    :param img_dir: image path
    :type img_dir: String
    :param feature_name: feature name
    :type feature_name: Features
    :return: True or False
    """
    feature_file_path = os.path.join(img_dir[:img_dir.find('videos')], 'features', 'Features_From_TUWien',
                                     'Image_Subtask', feature_name, img_dir[img_dir.find('videos') + 7:] + '.txt.gz')

    if os.path.isfile(feature_file_path):
        return True
    else:
        return False
