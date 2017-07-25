import numpy as np
import platform


def gen_submission_format(results):
    """
    reformats results for final submission
    :param results: the results. for each image path it contains the probability of class 'uninteresting'
    :type results: dict
    :return: videoname,keyframe-name,[classification decision: 1(interesting) or 0(not interesting)],[confidence value/interestingness level/rank]
    :rtype: np.array
    """

    submission_format = []
    for img_dir in results:
        if platform.system() == 'Linux':
            videoname = img_dir[img_dir.find('videos') + 7:img_dir.find('images') - 1]
            shotname = img_dir[img_dir.rfind('/') + 1:]
            classification = results[img_dir]['classification']
            confidence = results[img_dir]['probability']

            submission_format.append('{},{},{},{}'.format(videoname, shotname, classification, confidence))

            #raise NotImplementedError
        else: #windows
            videoname = img_dir[img_dir.find('videos')+7:img_dir.find('images')-1]
            shotname = img_dir[img_dir.rfind('\\')+1:]
            classification = results[img_dir]['classification']
            confidence = results[img_dir]['probability']

            submission_format.append('{},{},{},{}'.format(videoname, shotname, classification, confidence))

    return np.asarray(submission_format)