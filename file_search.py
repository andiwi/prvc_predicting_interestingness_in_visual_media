import os

def get_abs_path_of_file(directory, filename):
    '''
    search for a file with :filename in directory and all subdirectories of :directory
    :param directory:
    :param filename:
    :return: absolute path of file (of first occurrence) or None
    '''
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name == filename:
                # found file
                return os.path.abspath(os.path.join(root, name))

    return None