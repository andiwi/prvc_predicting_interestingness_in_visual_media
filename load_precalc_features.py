import scipy.io as sio
import os
import numpy as np
from read_imgs import read_img_names
from file_search import get_abs_path_of_file

def load_matlab_feature(path_to_imgs, feature_name):
    '''
    for all images in path_to_imgs directory: loads precalculated feature :feature_name
    :param path_to_imgs:
    :param feature_name:
    :return: numpy array of features of all imgs in path_to_imgs directory
    '''
    feature_root_dir = 'D:\\PR aus Visual Computing\\Interestingness17data\\features'

    img_names = read_img_names(path_to_imgs)

    features = []
    'D:\\PR aus Visual Computing\\Interestingness17data\\features\\'
    for img_name in img_names:

        if(feature_name == 'ColorHist'):
            feature = _colorHistogram_feature_extraction(feature_root_dir, img_name)
        else:
            raise NotImplementedError

        features.append(feature)

    return np.asarray(features)

def _colorHistogram_feature_extraction(feature_root_path, img_name):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'ColorHist')

    version = 1

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        version = 2
        # remove .jpg. extension in filename
        img_name_splitted = img_name.split('.')
        matfile_path = get_abs_path_of_file(path, img_name_splitted[0] + '.mat')

        if (matfile_path is None):
            version = -1
            print 'STOP'

    img_features = sio.loadmat(matfile_path)
    if (version == 1):
        return img_features['hsv'][0]
    if (version == 2):
        return img_features['ColorHist'][0]
    if (version == -1):
        raise NotImplementedError