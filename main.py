from draw_rule_of_third_lines import draw_rule_of_thirds_lines
from filecopy import filecopy
from draw_phi_grid import draw_phi_grid
from calc_histogram import calc_histograms, calc_histograms_plus_img, calc_histograms_normalized, calc_histograms_bw
from face_detection import face_detection, face_to_img_ratios_to_csv
from crop_imgs import crop_black_borders
import cv2
import os

def main():
    directory_original = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting'
    directory_cropped = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped'

#    filecopy()
#    crop_black_borders(directory_original)
#    draw_rule_of_thirds_lines(directory_cropped)
#    draw_phi_grid(directory_cropped)
#    calc_histograms(directory_cropped)
#    calc_histograms_normalized(directory_cropped)
#    calc_histograms_plus_img(directory_cropped)
#    calc_histograms_bw(directory_cropped)
#    face_detection(directory_cropped)
    face_to_img_ratios_to_csv(directory_cropped + '\\faces\\default')
if __name__ == '__main__':main()

