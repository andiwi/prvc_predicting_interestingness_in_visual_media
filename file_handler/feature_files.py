import os
import numpy as np

def save_features(dir, features):
    """
    saves features in files
    :param dir:
    :param features:
    :return:
    """
    for name in features:
        np.savetxt(os.path.join(dir, name + '_interesting.gz'), features[name][0])
        np.savetxt(os.path.join(dir, name + '_uninteresting.gz'), features[name][1])

def load_features(dir, feature_names):
    """
    loads features from file
    :param dir:
    :param feature_names:
    :return:
    """
    features = {}

    for name in feature_names:
        interesting = np.loadtxt(os.path.join(dir, name + '_interesting.gz'))
        uninteresting = np.loadtxt(os.path.join(dir, name + '_uninteresting.gz'))
        features[name] = [interesting, uninteresting]

    return features
