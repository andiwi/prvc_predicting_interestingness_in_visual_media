import os
import numpy as np
import gzip
from Features import Features
from feature_extraction.load_precalc_features import load_precalc_feature


def save_features(img_dir, feature_name, feature):
    """
    save features of img in .txt file (compressed in gzip format)
    :param img_dir: (string) path of the image
    :param feature_name: (string) name of the features (should be one of Features enumeration)
    :param feature: (numpy array) the features
    :return:
    """

    #create feature file path
    feature_file_path = os.path.join(img_dir[:img_dir.find('/videos/')], 'features', 'Features_From_TUWien', 'Image_Subtask', feature_name, img_dir[img_dir.find('/videos/')+8:] + '.txt.gz')
    dirs = feature_file_path[:feature_file_path.rfind('/')]
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    if feature_name == Features.Face_count \
        or feature_name == Features.Rot_distance:
        #feature is a single value
        with gzip.open(feature_file_path, "w") as f:
            f.write(str(feature))
    else:
        np.savetxt(feature_file_path, feature, newline=' ')

def load_features(img_dirs, feature_names):
    """
    loads features from file
    :param dir:
    :param feature_names:
    :return:
    """
    features = {}

    for img_dir in img_dirs:
        features[img_dir] = dict()
        for feature_name in feature_names:
            if Features.is_TU_feature(feature_name):
                feature = _load_TU_feature(img_dir, feature_name)
            else:
                feature = load_precalc_feature(img_dir, feature_name)

            features[img_dir][feature_name] = feature
    return features

def _load_TU_feature(img_dir, feature_name):
    feature_file_path = os.path.join(img_dir[:img_dir.find('/videos/')], 'features', 'Features_From_TUWien',
                                     'Image_Subtask', feature_name,
                                     img_dir[img_dir.find('/videos/') + 8:] + '.txt.gz')
    feature = np.loadtxt(feature_file_path)
    return feature
