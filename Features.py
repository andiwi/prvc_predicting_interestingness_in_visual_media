class Features(object):
    Face_count = 'face_count'
    Rot_distance = 'rot_distance'
    Face_bb = 'face_bb'
    Face_bb_full_img = 'face_bb_full_img'
    Face_bb_quarter_imgs = 'face_bb_quarter_imgs'
    Face_bb_eighth_imgs = 'face_bb_eighth_imgs'
    Tilted_edges = 'tilted_edges'
    Edge_hist_v0 = 'edge_hist_v0'
    Edge_hist_v1 = 'edge_hist_v1'
    Edge_hist_v2 = 'edge_hist_v2'
    Symmetry = 'symmetry'
    Hsv_hist = 'hsv_hist'
    DenseSIFT_L0 = 'denseSIFT_L0'
    DenseSIFT_L1 = 'denseSIFT_L1'
    DenseSIFT_L2 = 'denseSIFT_L2'
    Hog_L0 = 'hog_L0'
    Hog_L1 = 'hog_L1'
    Hog_L2 = 'hog_L2'
    Lbp_L0 = 'lbp_L0'
    Lbp_L1 = 'lbp_L1'
    Lbp_L2 = 'lbp_L2'
    Gist = 'gist'
    CNN_fc7 = 'cnn_fc7'
    CNN_prob = 'cnn_prob'

    @staticmethod
    def is_TU_feature(feature_name):
        if feature_name == Features.Face_count \
            or feature_name == Features.Rot_distance \
            or feature_name == Features.Face_bb \
            or feature_name == Features.Face_bb_full_img \
            or feature_name == Features.Face_bb_quarter_imgs \
            or feature_name == Features.Face_bb_eighth_imgs \
            or feature_name == Features.Tilted_edges \
            or feature_name == Features.Edge_hist_v0 \
            or feature_name == Features.Edge_hist_v1 \
            or feature_name == Features.Edge_hist_v2 \
            or feature_name == Features.Symmetry:
            return True
        else:
            return False

    @staticmethod
    def is_single_val_feature(features_name):
        if features_name == Features.Face_count \
            or features_name == Features.Rot_distance:
            return True
        else:
            return False

