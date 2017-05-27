import numpy as np

def distance_to_grid_corner(img, point):
    '''
    calculates euclidean distance from rule of thirds grid corner to point
    :param img: img read with cv2
    :param point: 2D numpy array
    :return: shortest euclidean distance
    '''
    height, width, channels = img.shape

    corner1 = np.array([height/3, width/3])
    corner2 = np.array([height/3, (width/3)*2])
    corner3 = np.array([(height/3)*2, width/3])
    corner4 = np.array([(height/3)*2, (width/3)*2])

    dist1 = np.linalg.norm(corner1-point)
    dist2 = np.linalg.norm(corner2-point)
    dist3 = np.linalg.norm(corner3-point)
    dist4 = np.linalg.norm(corner4-point)

    return np.minimum(np.minimum(dist1, dist2), np.minimum(dist3, dist4))