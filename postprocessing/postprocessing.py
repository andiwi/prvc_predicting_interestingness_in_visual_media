from file_handler.file_search import get_abs_path_of_file


def gen_submission_format(keyframe_name, int_levels):
    """
    reformats results for final submission
    :param img_names:
    :param int_levels:
    :return: np.array videoname,keyframe-name,[classification decision: 1(interesting) or 0(not interesting)],[confidence value/interestingness level/rank]
    """

    for i in range(0, len(keyframe_name)):
        #find video names
        path = get_abs_path_of_file(keyframe_name[i])

