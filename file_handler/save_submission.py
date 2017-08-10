import numpy as np

def save_submission(submission_format, runname):
    np.savetxt('me16in_groupname_image_{}.txt'.format(runname), submission_format, delimiter=',', fmt='%s')