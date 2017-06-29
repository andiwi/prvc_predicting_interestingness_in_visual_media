import os
import selectTrainingAndTestData
from preprocessing.crop_imgs import crop_black_borders

def preprocessing(dir):
    '''

    :param dir: root directory of images
    :return:
    '''
    dir_training_data = os.path.join(dir, 'trainingData')
    dir_test_data = os.path.join(dir, 'testData')

    # filecopy()
    selectTrainingAndTestData(os.path.join(dir, 'interesting'),
                              os.path.join(dir, 'uninteresting'), dir_training_data, dir_test_data,
                              interesting_training_samples=50, interesting_test_samples=50,
                              uninteresting_training_samples=500, uninteresting_test_samples=500)
    crop_black_borders(os.path.join(dir_training_data, 'interesting'))
    crop_black_borders(os.path.join(dir_training_data, 'uninteresting'))
    crop_black_borders(os.path.join(dir_test_data, 'interesting'))
    crop_black_borders(os.path.join(dir_test_data, 'uninteresting'))