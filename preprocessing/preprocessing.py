from crop_imgs import crop_black_borders

def preprocessing(img_dirs):
    """
    does some preprocessing for each image of img_dirs. use this before feature calculation. only necessary on new
    datasets
    :param img_dirs: a list of image paths
    :type img_dirs: list
    :return:
    :rtype:
    """
    crop_black_borders(img_dirs)
