import numpy as np

def numpy_fillwithzeros(data):
    '''
    fills rows with less columns with zeros so that all rows have same number of columns
    :param data: 
    :return: 
    '''
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

def numpy_fillcolswithzeros(array1, array2):
    '''
    find array with more columns and fill missing array columns with zeros of array with less columns
    :param array1: 
    :param array2: 
    :return: reshaped array1 and array2
    '''
    r, c = array1.shape
    r2, c2 = array2.shape
    if c > c2:
        zeros = np.zeros((r2, c - c2))
        array2 = np.concatenate((array2, zeros), axis=1)
    else:
        zeros = np.zeros((r, c2 - c))
        array1 = np.concatenate((array1, zeros), axis=1)

    return array1, array2

def numpy_extend_cols_array_w_zeros(array, zero_count):
    '''
    fill array columns with zeros
    :param array: (numpy array) the columns will be filled with zeros
    :param zero_count: (int) the amount of zeros which should be added
    :return: reshaped array
    '''
    r, c = array.shape
    zeros = np.zeros((r, zero_count))
    array = np.concatenate((array, zeros), axis=1)

    return array
