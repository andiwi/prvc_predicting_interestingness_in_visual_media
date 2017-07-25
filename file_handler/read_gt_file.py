import os
from collections import OrderedDict


def read_img_dirs_and_gt(dir):
    """
    read all image paths and interestingness level from ground truth file
    :param dir: root directory of dataset e.x. './InterestingnessData16/devset/' or './InterestingnessData16/testset/'
    :type dir: str
    :return: (img_path, gt)
    :rtype: dict
    """
    gt_path = os.path.join(dir, 'annotations', 'devset-image.txt')

    # read ground truth file
    with open(gt_path) as f:
        gt_content = f.readlines()
        gt_content = [x.strip() for x in gt_content]  # remove whitespace characters like `\n` at the end of each line
        gt_content = [x.split(',') for x in gt_content]  # split by delimiter ','

    img_dirs_gt = OrderedDict()

    for row in gt_content:
        img_dir = os.path.join(dir, 'videos', row[0], 'images', row[1])
        img_gt = row[2]

        img_dirs_gt[img_dir] = img_gt

    return img_dirs_gt
