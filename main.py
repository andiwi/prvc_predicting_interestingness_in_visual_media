from draw_rule_of_third_lines import draw_rule_of_thirds_lines
from filecopy import filecopy
from draw_phi_grid import draw_phi_grid
from calc_histogram import calc_histograms, calc_histograms_plus_img, calc_histograms_normalized, calc_histograms_bw
from face_detection import face_detection, face_to_img_ratios_to_csv
from crop_imgs import crop_black_borders
from selectTrainingAndTestData import selectTrainingAndTestData
import cv2
import os

def main():
    directory_original = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting'
    directory_cropped = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped'

    directory_root = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images'

    #preprocessing
    selectTrainingAndTestData(os.path.join(directory_root, 'interesting'), os.path.join(directory_root, 'uninteresting'), os.path.join(directory_root, 'trainingData'), os.path.join(directory_root, 'testData'))
    #calculate features

    #train svm

    #test svm





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
