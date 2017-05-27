from draw_rule_of_third_lines import draw_rule_of_thirds_lines
from filecopy import filecopy
from draw_phi_grid import draw_phi_grid
from calc_histogram import calc_histograms, calc_histograms_plus_img, calc_histograms_normalized, calc_histograms_bw
from face_detection import face_detection, face_to_img_ratios_to_csv
from crop_imgs import crop_black_borders
from selectTrainingAndTestData import selectTrainingAndTestData
from read_imgs import read_img_names
from sklearn import svm, datasets
import numpy as np
import np_helper
import matplotlib.pyplot as plt
import cv2
import os

def main():
    preprocessing = False
    calc_features = True
    load_features = True

    directory_original = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting'
    directory_cropped = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped'

    directory_root = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images'
    dir_training_data = os.path.join(directory_root, 'trainingData')
    dir_test_data = os.path.join(directory_root, 'testData')

    #
    #preprocessing
    #
    if(preprocessing):
        selectTrainingAndTestData(os.path.join(directory_root, 'interesting'), os.path.join(directory_root, 'uninteresting'), dir_training_data, dir_test_data, interesting_training_samples=10, interesting_test_samples=10, uninteresting_training_samples=100, uninteresting_test_samples=100)
        crop_black_borders(os.path.join(dir_training_data, 'interesting'))
        crop_black_borders(os.path.join(dir_training_data, 'uninteresting'))
        crop_black_borders(os.path.join(dir_test_data, 'interesting'))
        crop_black_borders(os.path.join(dir_test_data, 'uninteresting'))

    #img_names = read_img_names(dir_training_data)

    #
    #calculate features
    #
    if(calc_features):
        #calc face features
        face_features_interesting = face_detection(os.path.join(dir_training_data, 'interesting'))
        face_features_uninteresting = face_detection(os.path.join(dir_training_data, 'uninteresting'))

        face_features_interesting_test = face_detection(os.path.join(dir_test_data, 'interesting'))
        face_features_uninteresting_test = face_detection(os.path.join(dir_test_data, 'uninteresting'))


        #
        # save features
        #
        np.savetxt(os.path.join(dir_training_data, 'face_features_interesting.gz'), face_features_interesting)
        np.savetxt(os.path.join(dir_training_data, 'face_features_uninteresting.gz'), face_features_uninteresting)

        np.savetxt(os.path.join(dir_test_data, 'face_features_interesting.gz'), face_features_interesting_test)
        np.savetxt(os.path.join(dir_test_data, 'face_features_uninteresting.gz'), face_features_uninteresting_test)

    if(load_features):
        #
        # load features from files
        #
        face_features_interesting = np.loadtxt(os.path.join((dir_training_data), 'face_features_interesting.gz'))
        face_features_uninteresting = np.loadtxt(os.path.join((dir_training_data), 'face_features_uninteresting.gz'))

        face_features_interesting_test = np.loadtxt(os.path.join((dir_test_data), 'face_features_interesting.gz'))
        face_features_uninteresting_test = np.loadtxt(os.path.join((dir_test_data), 'face_features_uninteresting.gz'))

    #
    # concatenate features
    #

    #find matrix with maximal columns and reshape other matrixe before concatenating them
    face_features_interesting, face_features_uninteresting = np_helper_fillcolswithzeros(face_features_interesting, face_features_uninteresting)
    face_features = np.concatenate((face_features_interesting, face_features_uninteresting), axis=0)

    face_features_interesting_test, face_features_uninteresting_test = np_helper.fillcolswithzeros(face_features_interesting_test, face_features_uninteresting_test)
    face_features_test = np.concatenate((face_features_interesting_test, face_features_uninteresting_test), axis=0)

    #reshape, so that training and test set have same number of columns
    face_features, face_features_test = np_helper.numpy_fillcolswithzeros(face_features, face_features_test)

    #
    # get interestingness
    #
    img_names_interesting = read_img_names(os.path.join(dir_training_data, 'interesting'))
    img_names_uninteresting = read_img_names(os.path.join(dir_training_data, 'uninteresting'))
    target_interesting = np.ones((len(img_names_interesting),1))
    target_uninteresting = np.zeros((len(img_names_uninteresting), 1))

    img_names_interesting_test = read_img_names(os.path.join(dir_test_data, 'interesting'))
    img_names_uninteresting_test = read_img_names(os.path.join(dir_test_data, 'uninteresting'))
    target_interesting_test = np.ones((len(img_names_interesting_test), 1))
    target_uninteresting_test = np.zeros((len(img_names_uninteresting_test), 1))

    target = np.concatenate((target_interesting, target_uninteresting), axis=0)
    target_test = np.concatenate((target_interesting_test, target_uninteresting_test), axis=0)


    #
    # train svm
    #
    X = face_features
    y = target

    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C)
    svc.fit(X, y)

    #
    # test svm
    #
    X_test = face_features_test
    y_pred = svc.predict(X_test)
    '''
    #DEBUG EXAMPLE
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
    '''


    #
    # #test svm
    #




#    filecopy()
#    crop_black_borders(directory_original)
#    draw_rule_of_thirds_lines(directory_cropped)
#    draw_phi_grid(directory_cropped)
#    calc_histograms(directory_cropped)
#    calc_histograms_normalized(directory_cropped)
#    calc_histograms_plus_img(directory_cropped)
#    calc_histograms_bw(directory_cropped)

#    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
#    face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
#    face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')
#    #face_detection(directory_cropped, face_frontal_cascade, face_profile_cascade)
#    face_to_img_ratios_to_csv(directory_cropped + '\\faces\\', face_frontal_cascade, face_profile_cascade)

if __name__ == '__main__':main()

