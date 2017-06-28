import os
import random
from shutil import copyfile

from file_handler.read_imgs import read_img_names


def selectTrainingAndTestData(dir_interesting_imgs, dir_uninteresting_imgs, output_dir_training, output_dir_test, interesting_training_samples=50, uninteresting_training_samples=500, interesting_test_samples=50, uninteresting_test_samples=500):
    """
    selects randomly interesting and uninteresting imgs and stores them at output_directory 
    
    :param dir_interesting_imgs: directory of all interesting images
    :param dir_uninteresting_imgs: directory of all uninteresting images
    :param output_dir_training: output directory for training data
    :param output_dir_test: output directory for test data
    :param interesting_training_samples: number of how much interesting imgs should be selected for training data set
    :param uninteresting_training_samples: number of how much uninteresting imgs should be selected for training data set
    :param interesting_test_samples: number of how much interesting imgs should be selected for test data set
    :param uninteresting_test_samples: number of how much uninteresting imgs should be selected for test data set
    :return: 
    """

    #select random img_names
    #interesting
    img_names_interesting = read_img_names(dir_interesting_imgs)
    total_sample_number_interesting = interesting_training_samples + interesting_test_samples

    rnd_interesting_img_names = random.sample(img_names_interesting, total_sample_number_interesting)
    rnd_interesting_img_names_training = rnd_interesting_img_names[:interesting_training_samples]
    rnd_interesting_img_names_test = rnd_interesting_img_names[interesting_training_samples:total_sample_number_interesting]

    if not os.path.exists(os.path.join(output_dir_training,'interesting')):
        os.makedirs(os.path.join(output_dir_training,'interesting'))

    if not os.path.exists(os.path.join(output_dir_test,'interesting')):
        os.makedirs(os.path.join(output_dir_test,'interesting'))

    for img_name in rnd_interesting_img_names_training:
        copyfile(os.path.join(dir_interesting_imgs, img_name), os.path.join(output_dir_training,'interesting',img_name))

    for img_name in rnd_interesting_img_names_test:
        copyfile(os.path.join(dir_interesting_imgs, img_name), os.path.join(output_dir_test,'interesting',img_name))

    #uninteresting
    img_names_uninteresting = read_img_names(dir_uninteresting_imgs)
    total_sample_number_uninteresting = uninteresting_training_samples + uninteresting_test_samples

    rnd_uninteresting_img_names = random.sample(img_names_uninteresting,total_sample_number_uninteresting)
    rnd_uninteresting_img_names_training = rnd_uninteresting_img_names[:uninteresting_training_samples]
    rnd_uninteresting_img_names_test = rnd_uninteresting_img_names[uninteresting_training_samples:total_sample_number_uninteresting]

    if not os.path.exists(os.path.join(output_dir_training, 'uninteresting')):
        os.makedirs(os.path.join(output_dir_training, 'uninteresting'))

    if not os.path.exists(os.path.join(output_dir_test, 'uninteresting')):
        os.makedirs(os.path.join(output_dir_test, 'uninteresting'))

    for img_name in rnd_uninteresting_img_names_training:
        copyfile(os.path.join(dir_uninteresting_imgs, img_name),
                 os.path.join(output_dir_training, 'uninteresting', img_name))

    for img_name in rnd_uninteresting_img_names_test:
        copyfile(os.path.join(dir_uninteresting_imgs, img_name), os.path.join(output_dir_test, 'uninteresting', img_name))


