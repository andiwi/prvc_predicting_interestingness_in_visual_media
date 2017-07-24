import numpy as np

def save_submission(submission_format):
    np.savetxt('me17in_groupname_image_1.txt', submission_format, delimiter=',', fmt='%s')