from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from helper import box_filter


def predict(svc, X_train, y_train, X_test, features_test, use_second_dev_classification_method):
    """
    trains SVM and predicts classes for X_test

    :param svc: the SVM
    :type svc: sklearn.SVC
    :param X_train: numpy array of feature vectors
    :type X_train: np.array
    :param y_train: numpy array of target classes
    :type y_train: np.array
    :param X_test: numpy array of feature vectors
    :type X_test: np.array
    :param features_test:
    :type features_test: dict
    :param use_second_dev_classification_method: if second order classification method should be used
    :type use_second_dev_classification_method: bool
    :return: results
    :rtype:
    """

    # train svm
    svc.fit(X_train, y_train)
    # classify test set
    y_scores = svc.decision_function(X_test)
    y_predicted = svc.predict(X_test)

    if np.ptp(y_scores) != 0:
        # Normalised [0,1]
        y_scores = (y_scores - np.max(y_scores)) / -np.ptp(y_scores)

    # random
    # y_scores = np.random.uniform(0, 1, size=(2342))
    # y_predicted = y_scores > 0.5
    # y_predicted = y_predicted.astype(int)

    # reassign probabilities and classification to images
    counter = 0
    results = OrderedDict()
    for img_dir in features_test.keys():
        results[img_dir] = OrderedDict()
        results[img_dir]['probability'] = y_scores[counter]
        results[img_dir]['classification'] = y_predicted[counter]
        counter = counter + 1

    # calc final classification
    if use_second_dev_classification_method:
        # order scores, calc threshold and overwrite classification result

        probability = y_scores
        x_vals = np.asarray(range(0, len(probability)))

        # sort probability
        probability = np.sort(probability)

        # smooth curve with averaging window
        proba_smooth = box_filter.smooth(probability, 3)

        # calc second order derivative
        diff_2nd = np.diff(proba_smooth, 2)

        # find first point above threshold
        thresh = 0.01
        limitIdx = None  # This position corresponds to the limit between non interesting and interesting shots/key-frames.
        for i in range(2, len(diff_2nd)):
            if diff_2nd[i] > thresh:
                limitIdx = i
                break

        if limitIdx is None:
            limitIdx = i

        # DEBUG
        plt.plot(x_vals, probability, 'o')
        plt.plot(x_vals, proba_smooth, 'r-', lw=2)
        plt.plot(x_vals[limitIdx], proba_smooth[limitIdx], 'go')
        plt.show()
        # DEBUG END

        limitProba = proba_smooth[limitIdx]

        for img_dir in results:
            if results[img_dir]['probability'] > limitProba:
                results[img_dir]['classification'] = 1
            else:
                results[img_dir]['classification'] = 0

    return results