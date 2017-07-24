import os
import cv2
import numpy as np

from face_detection import detect_faces
from feature_extraction.load_precalc_features import load_precalc_feature
from file_handler.feature_files import TU_feature_file_exists
from file_handler.read_imgs import read_img
from mpeg7_edge_histogram import calc_edge_histogram
from face_detection import face_detection
from chainercv.links import FasterRCNNVGG16
from Features import Features
from helper.Geometry import Rect
from file_handler import feature_files


def calc_features(img_dirs, feature_names):
    """
    calculates all features given by feature_names
    :param img_dirs: a list of image paths
    :type img_dirs: list
    :param feature_names: a list containing all feature names
    :type feature_names: list
    :return: dict (img_dir, features) features is also a dict (feature_name, feature vector)
    :rtype: dict
    """

    features = {}

    dir_haarfeatures = os.path.join(os.getcwd(), 'res', 'haarcascades')
    face_frontal_cascade = cv2.CascadeClassifier(os.path.join(dir_haarfeatures, 'haarcascade_frontalface_default.xml'))
    face_profile_cascade = cv2.CascadeClassifier(os.path.join(dir_haarfeatures, 'haarcascade_profileface.xml'))

    model = FasterRCNNVGG16(pretrained_model='voc07')

    for img_dir in img_dirs:
        print 'calc features for: ' + img_dir
        img = read_img(img_dir)
        features[img_dir] = dict() #init features dict for each image

        for name in feature_names:
            if TU_feature_file_exists(img_dir, name):
                continue

            if name == Features.Face_count or name == Features.Rot_distance or name == Features.Face_bb:
                face_count, rot_distance, face_bb = face_detection(img, face_frontal_cascade, face_profile_cascade)

                feature_files.save_features(img_dir, Features.Face_count, face_count)
                feature_files.save_features(img_dir, Features.Rot_distance, rot_distance)
                feature_files.save_features(img_dir, Features.Face_bb, face_bb)

                features[img_dir][Features.Face_count] = face_count
                features[img_dir][Features.Rot_distance] = rot_distance
                features[img_dir][Features.Face_bb] = face_bb

            elif name == Features.Face_bb_full_img:
                feature = _face_bb_full_img_calculator(img, face_frontal_cascade, face_profile_cascade)
                feature_files.save_features(img_dir, Features.Face_bb_full_img, feature)

                features[img_dir][Features.Face_bb_full_img] = feature

            elif name == Features.Face_bb_quarter_imgs:
                feature = _face_bb_quarter_imgs_calculator(img, face_frontal_cascade, face_profile_cascade)
                feature_files.save_features(img_dir, Features.Face_bb_quarter_imgs, feature)

                features[img_dir][Features.Face_bb_quarter_imgs] = feature

            elif name == Features.Face_bb_eighth_imgs:
                feature = _face_bb_eighth_imgs_calculator(img, face_frontal_cascade, face_profile_cascade)
                feature_files.save_features(img_dir, Features.Face_bb_eighth_imgs, feature)

                features[img_dir][Features.Face_bb_eighth_imgs] = feature

            elif name == Features.Tilted_edges:
                feature = _img_tilted_calculator(img)
                feature_files.save_features(img_dir, Features.Tilted_edges, feature)

                features[img_dir][Features.Tilted_edges] = feature

            elif name == Features.Edge_hist_v0:
                feature = _edge_hist_dir_calculator(img, False, False)
                feature_files.save_features(img_dir, Features.Edge_hist_v0, feature)

                features[img_dir][Features.Edge_hist_v0] = feature

            elif name == Features.Edge_hist_v1:
                feature = _edge_hist_dir_calculator(img, True, True)
                feature_files.save_features(img_dir, Features.Edge_hist_v1, feature)

                features[img_dir][Features.Edge_hist_v1] = feature

            elif name == Features.Edge_hist_v2:
                feature = _edge_hist_dir_calculator(img, True, False)
                feature_files.save_features(img_dir, Features.Edge_hist_v2, feature)

                features[img_dir][Features.Edge_hist_v2] = feature

            elif name == Features.Symmetry:
                feature = _symmetry_calculator(img, model)
                feature_files.save_features(img_dir, Features.Symmetry, feature)

                features[img_dir][Features.Symmetry] = feature

            # precalculated features
            elif name == Features.Hsv_hist:
                feature = load_precalc_feature(img_dir, Features.Hsv_hist)
                feature_files.save_features(img_dir, Features.Hsv_hist, feature)

                features[img_dir][Features.Hsv_hist] = feature

            elif name == Features.DenseSIFT_L0:
                feature = load_precalc_feature(img_dir, Features.DenseSIFT_L0)
                feature_files.save_features(img_dir, Features.DenseSIFT_L0, feature)

                features[img_dir][Features.DenseSIFT_L0] = feature

            elif name == Features.DenseSIFT_L1:
                feature = load_precalc_feature(img_dir, Features.DenseSIFT_L1)
                feature_files.save_features(img_dir, Features.DenseSIFT_L1, feature)

                features[img_dir][Features.DenseSIFT_L1] = feature

            elif name == Features.DenseSIFT_L2:
                feature = load_precalc_feature(img_dir, Features.DenseSIFT_L2)
                feature_files.save_features(img_dir, Features.DenseSIFT_L2, feature)

                features[img_dir][Features.DenseSIFT_L2] = feature

            elif name == Features.Hog_L0:
                feature = load_precalc_feature(img_dir, Features.Hog_L0)
                feature_files.save_features(img_dir, Features.Hog_L0, feature)

                features[img_dir][Features.Hog_L0] = feature

            elif name == Features.Hog_L1:
                feature = load_precalc_feature(img_dir, Features.Hog_L1)
                feature_files.save_features(img_dir, Features.Hog_L1, feature)

                features[img_dir][Features.Hog_L1] = feature

            elif name == Features.Hog_L2:
                feature = load_precalc_feature(img_dir, Features.Hog_L2)
                feature_files.save_features(img_dir, Features.Hog_L2, feature)

                features[img_dir][Features.Hog_L2] = feature

            elif name == Features.Lbp_L0:
                feature = load_precalc_feature(img_dir, Features.Lbp_L0)
                feature_files.save_features(img_dir, Features.Lbp_L0, feature)

                features[img_dir][Features.Lbp_L0] = feature

            elif name == Features.Lbp_L1:
                feature = load_precalc_feature(img_dir, Features.Lbp_L1)
                feature_files.save_features(img_dir, Features.Lbp_L1, feature)

                features[img_dir][Features.Lbp_L1] = feature

            elif name == Features.Lbp_L2:
                feature = load_precalc_feature(img_dir, Features.Lbp_L2)
                feature_files.save_features(img_dir, Features.Lbp_L2, feature)

                features[img_dir][Features.Lbp_L2] = feature

            elif name == Features.Gist:
                feature = load_precalc_feature(img_dir, Features.Gist)
                feature_files.save_features(img_dir, Features.Gist, feature)

                features[img_dir][Features.Gist] = feature

            elif name == Features.CNN_fc7:
                feature = load_precalc_feature(img_dir, Features.CNN_fc7)
                feature_files.save_features(img_dir, Features.CNN_fc7, feature)

                features[img_dir][Features.CNN_fc7] = feature

            elif name == Features.CNN_prob:
                feature = load_precalc_feature(img_dir, Features.CNN_prob)
                feature_files.save_features(img_dir, Features.CNN_prob, feature)

                features[img_dir][Features.CNN_prob] = feature

            else:
                raise NotImplementedError

    return features


def _calc_features(img_dirs, feature_calculator, *args):
    #DEBUG
    # counter = 0
    #DEBUG END
    features = []
    for img_dir in img_dirs:
        img = read_img(img_dir)

        if args is None:
            feature = feature_calculator(img)
        else:
            feature = feature_calculator(img, *args)

        feature_files.save_features(img_dir, feature)
        features.append(feature)
        #DEBUG
        # print 'image: ' + str(counter)
        # counter = counter+1
        #DEBUG END
    return np.asarray(features)


def _face_count_calculator(img, face_frontal_cascade=None, face_profile_cascade=None):
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    if face_frontal_cascade is None:
        face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')

    if face_profile_cascade is None:
        face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(img, face_frontal_cascade, face_profile_cascade)

    return len(rect_faces)


def _img_tilted_calculator(img):
    '''
    uses edge histogram to detect if camera perspective is tilted
    :param img:
    :return: percentage of edges that are tilted
    '''
    hists, quant_hist, global_hist, quant_global_hist, semiglobal_hist, quant_semiglobal_hist = calc_edge_histogram(img)

    # not_tilted = sum(global_hist[0:2])
    tilted = sum(global_hist[2:4])
    # non_directional_edges = global_hist[4]

    return tilted / sum(global_hist)


def _edge_hist_dir_calculator(img, only_strongest_dir=False, maintain_values=True):
    '''
    calculates edge histogram, sets
    :param img:
    :param only_strongest_dir if True -> only the strongest direction of a block has a value. Other directions are set to 0
    :param maintain_values if False -> values of edge histogram were replaced by a 1 for the max direction
    :return: edge histogram in a row as a feature vector for the image
    '''
    hists, quant_hist, global_hist, quant_global_hist, semiglobal_hist, quant_semiglobal_hist = calc_edge_histogram(img)

    if only_strongest_dir:
        # for every block find maximum value and set other values to 0
        def setToZeros(row):
            max_val = row.max()
            row = np.where(row != max_val, 0, max_val)
            return row

        hists = np.apply_along_axis(setToZeros, axis=1, arr=hists)

    if not maintain_values:
        # set remaining values to 1
        hists = np.where(hists > 0, 1, 0)

    hists_1D = hists.flatten()

    return hists_1D


def _face_bb_full_img_calculator(img, face_frontal_cascade, face_profile_cascade):
    """
    detects faces in image. returns biggest bounding box

    :param img:
    :param face_frontal_cascade:
    :param face_profile_cascade:
    :return: the biggest bounding boxes
    """
    # DEBUG
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # DEBUG END

    #rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(img)
    rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(img, face_frontal_cascade, face_profile_cascade)
    # sort list in descending order
    rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

    if len(rect_faces) > 0:
        bbox = [rect_faces[0].x,
                rect_faces[0].y,
                rect_faces[0].w,
                rect_faces[0].h]
    else:
        bbox = [0, 0, 0, 0]

    return np.asarray(bbox)


def _face_bb_quarter_imgs_calculator(img, face_frontal_cascade, face_profile_cascade):
    """
    splits the image into 4 subimages. for each subimage it detects faces, chooses the face with biggest bounding box and stores bb (x,y,w,h) as feature

    How the image is splitted:
     _________
    |    |    |
    |__1_|__2_|
    |    |    |
    |__3_|_4__|

    :param img:
    :param face_frontal_cascade:
    :param face_profile_cascade:
    :return: the biggest bounding boxes (1 for each subimg)
    """

    # split image into 4 quarter images
    height, width, channels = img.shape

    subimgs = [img[0:int(height / 2), 0:int(width / 2)],
               img[0:int(height / 2), int(width / 2):width],
               img[int(height / 2):height, 0:int(width / 2)],
               img[int(height / 2):height, int(width / 2):width]
               ]

    bboxes = []

    # DEBUG
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    #
    # for subimg in subimgs:
    #    cv2.imshow('image', subimg)
    #    cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    # DEBUG END

    for subimg in subimgs:
        rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(subimg, face_frontal_cascade,
                                                                          face_profile_cascade)
        # sort list in descending order
        rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

        if len(rect_faces) > 0:

            bboxes.append(rect_faces[0].x)
            bboxes.append(rect_faces[0].y)
            bboxes.append(rect_faces[0].w)
            bboxes.append(rect_faces[0].h)
        else:
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)

    return np.asarray(bboxes)


def _face_bb_eighth_imgs_calculator(img, face_frontal_cascade, face_profile_cascade):
    """
    splits the image into 8 subimages. for each subimage it detects faces, chooses the face with biggest bounding box and stores bb (x,y,w,h) as feature

    How the image is splitted:

    :param img:
    :param face_frontal_cascade:
    :param face_profile_cascade:
    :return: the biggest bounding boxes (1 for each subimg)
    """

    # split image into 4 quarter images
    height, width, channels = img.shape

    subimgs = [img[0:int(height / 4), 0:int(width / 4)],
               img[0:int(height / 4), int(width / 4):int(width / 2)],
               img[0:int(height / 4), int(width / 2):int(width / 4) * 3],
               img[0:int(height / 4), int(width / 4) * 3:width],

               img[int(height / 4):int(height / 2), 0:int(width / 4)],
               img[int(height / 4):int(height / 2), int(width / 4):int(width / 2)],
               img[int(height / 4):int(height / 2), int(width / 2):int(width / 4) * 3],
               img[int(height / 4):int(height / 2), int(width / 4) * 3:width],

               img[int(height / 2):int(height / 4) * 3, 0:int(width / 4)],
               img[int(height / 2):int(height / 4) * 3, int(width / 4):int(width / 2)],
               img[int(height / 2):int(height / 4) * 3, int(width / 2):int(width / 4) * 3],
               img[int(height / 2):int(height / 4) * 3, int(width / 4) * 3:width],

               img[int(height / 4) * 3:height, 0:int(width / 4)],
               img[int(height / 4) * 3:height, int(width / 4):int(width / 2)],
               img[int(height / 4) * 3:height, int(width / 2):int(width / 4) * 3],
               img[int(height / 4) * 3:height, int(width / 4) * 3:width]
               ]

    bboxes = []

    # DEBUG
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    #
    # for subimg in subimgs:
    #    cv2.imshow('image', subimg)
    #    cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    # DEBUG END

    for subimg in subimgs:
        rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(subimg, face_frontal_cascade,
                                                                          face_profile_cascade)
        # sort list in descending order
        rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

        if len(rect_faces) > 0:

            bboxes.append(rect_faces[0].x)
            bboxes.append(rect_faces[0].y)
            bboxes.append(rect_faces[0].w)
            bboxes.append(rect_faces[0].h)
        else:
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)

    return np.asarray(bboxes)


def _symmetry_calculator(img, model):
    height, width, channels = img.shape

    # split image into left and right image
    # img_l = [:, 0:int(width/2), :]
    # img_r = [:, int(width/2):width,:]

    # convert image
    img_converted = np.empty((channels, height, width))
    img_converted[0, :, :] = img[:, :, 2]
    img_converted[1, :, :] = img[:, :, 1]
    img_converted[2, :, :] = img[:, :, 0]

    img_left = img_converted[:, :, 0:int(width / 2)]
    img_right = img_converted[:, :, int(width / 2):width]

    bboxes_l, labels_l, scores_l = model.predict([img_left])
    bboxes_r, labels_r, scores_r = model.predict([img_right])

    bboxes_l = bboxes_l[0]
    labels_l = labels_l[0]
    scores_l = scores_l[0]

    bboxes_r = bboxes_r[0]
    labels_r = labels_r[0]
    scores_r = scores_r[0]

    # DEBUG
    # vis_bbox(
    #    img_left, bboxes_l, labels_l, scores_l, label_names=voc_detection_label_names)
    # plot.show()
    #
    # vis_bbox(
    #    img_right, bboxes_r, labels_r, scores_r, label_names=voc_detection_label_names)
    # plot.show()
    # DEBUG END

    if(bboxes_l.size == 0 or bboxes_r.size == 0):
        #no objects found in image
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # store bbox labels and score in dict
    descr_l = {}
    for i in range(len(bboxes_l)):
        bbox = bboxes_l[i]
        label = labels_l[i]
        score = scores_l[i]

        rect = Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        descr_l[rect] = [label, score]

    descr_r = {}
    for i in range(len(bboxes_r)):
        bbox = bboxes_r[i]
        label = labels_r[i]
        score = scores_r[i]

        rect = Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        descr_r[rect] = [label, score]
    # sort
    rects_sorted_l = sorted(descr_l, key=lambda rect: rect.area(), reverse=True)
    rects_sorted_r = sorted(descr_r, key=lambda rect: rect.area(), reverse=True)

    match_found = False

    # first try: select biggest obj in left image and search for same obj in right image
    # find biggest bboxes in left image
    sel_bbox_l = rects_sorted_l[0]
    sel_label_l = descr_l[sel_bbox_l][0]
    sel_score_l = descr_l[sel_bbox_l][1]

    # check if in other img is a bbox with same label
    for i in range(len(descr_r)):
        [label, score] = descr_r[rects_sorted_r[i]]
        if label == sel_label_l:
            # found match
            sel_bbox_r = rects_sorted_r[i]
            sel_label_r = label
            sel_score_r = score
            match_found = True
            break

    if not match_found:
        # second try: select biggest obj in right image and search for same obj in left image
        # find biggest bboxes in left image
        sel_bbox_r = rects_sorted_r[0]
        sel_label_r = descr_r[sel_bbox_r][0]
        sel_score_r = descr_r[sel_bbox_r][1]

        # check if in other img is a bbox with same label
        for i in range(len(descr_l)):
            [label, score] = descr_l[rects_sorted_l[i]]
            if label == sel_label_r:
                # found match
                sel_bbox_l = rects_sorted_l[i]
                sel_label_l = label
                sel_score_l = score
                match_found = True
                break

    # return label, score_l, score_r, bbox_l, bbox_r
    if match_found:
        return [sel_bbox_l.x, sel_bbox_l.y, sel_bbox_l.w, sel_bbox_l.h, sel_label_l, sel_score_l, sel_bbox_r.x,
                sel_bbox_r.y, sel_bbox_r.w, sel_bbox_r.h, sel_label_r, sel_score_r]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
