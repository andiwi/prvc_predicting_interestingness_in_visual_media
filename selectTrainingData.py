from shutil import copyfile
import os
import random
from read_imgs import read_img_names, read_img

def selectTrainingsData(directory_interesting_imgs, directory_uninteresting_imgs, interesting_number, uninteresting_number, directory_output):
    """
    selects randomly interesting and uninteresting imgs and stores them at output_directory
    
    :param directory_interesting_imgs: directory of all interesting images
    :param directory_uninteresting_imgs: directory of all uninteresting images
    :param interestingNumber: how much interesting imgs should be selected
    :param uninterestingNumber: how much uninteresting imgs should be selected
    :return: 
    """
    #select random img_names
    img_names_interesting = read_img_names(directory_interesting_imgs)
    rand_interesting_img_names = random.sample(img_names_interesting, interesting_number)

    if not os.path.exists(os.path.join(directory_output,'interesting')):
        os.makedirs(os.path.join(directory_output,'interesting'))

    if not os.path.exists(os.path.join(directory_output,'uninteresting')):
        os.makedirs(os.path.join(directory_output,'uninteresting'))

    for img_name in rand_interesting_img_names:
        copyfile(os.path.join(directory_interesting_imgs, img_name), os.path.join(directory_output,'interesting',img_name))

    img_names_uninteresting = read_img_names(directory_uninteresting_imgs)
    rand_interesting_img_names = random.sample(img_names_uninteresting, uninteresting_number)

    for img_name in rand_interesting_img_names:
        copyfile(os.path.join(directory_uninteresting_imgs, img_name), os.path.join(directory_output,'uninteresting',img_name))


