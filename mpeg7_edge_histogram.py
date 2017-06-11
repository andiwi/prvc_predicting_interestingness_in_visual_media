#adapted from source: https://github.com/tftdias/mp7descriptors/blob/master/edge_histogram_descriptor_generator.py
__author__ = 'Tiago'

import cv2
import numpy as np
import sys

def calc_edge_histogram(img):
    '''
    Calculates the mpeg7 edge histogram according to https://www.dcc.fc.up.pt/~mcoimbra/lectures/VC_1415/VC_1415_P8_LEH.pdf
    :param img: the img
    :return: hists, quant_hist, global_hist, quant_global_hist, semiglobal_hist, quant_semiglobal_hist
    '''
    DIRECTION_FILTERS = np.array([[1, -1, 1, -1],  # vertical
                                  [1, 1, -1, -1],  # horizontal
                                  [np.sqrt(2), 0, 0, -np.sqrt(2)],  # 45 diagonal
                                  [0, np.sqrt(2), -np.sqrt(2), 0],  # 135 diagonal
                                  [2, -2, -2, 2]  # nao direccional
                                  ]).T
    DIRECTION_THRES = 0

    QUANTIZER_MATRIX = np.array([[0.010867, 0.012266, 0.004193, 0.004174, 0.006778],
                                 [0.057915, 0.069934, 0.025852, 0.025924, 0.051667],
                                 [0.099526, 0.125879, 0.046860, 0.046232, 0.108650],
                                 [0.144849, 0.182307, 0.068519, 0.067163, 0.166257],
                                 [0.195573, 0.243396, 0.093286, 0.089655, 0.224226],
                                 [0.260504, 0.314563, 0.123490, 0.115391, 0.285691],
                                 [0.358031, 0.411728, 0.161505, 0.151904, 0.356375],
                                 [0.530128, 0.564319, 0.228960, 0.217745, 0.450972]])


    img = img

    hists = np.zeros([16, 5])

    subimg_height = img.shape[0] / 4
    subimg_width = img.shape[1] / 4

    desired_num_block = 10
    block_size = int(np.floor(np.sqrt(subimg_height * subimg_width / desired_num_block) / 2) * 2)
    block_width = block_size
    block_height = block_size
    blocks_per_subimg = subimg_height / block_height * subimg_width / block_width

    subimg_index = 0
    block_index = 0

    for i in np.arange(0, img.shape[0], subimg_height):
        for j in np.arange(0, img.shape[1], subimg_width):
            if subimg_index is 16:
                #this is the case if subimg_height or subimg_width is not divideable by 4. Therefore we ignore the small border.
                break

            subimg = img[i:i + subimg_height, j:j + subimg_width]
            #print i, i + subimg_height
            #print j, j + subimg_width
            # DEBUG
            #cv2.namedWindow('Subimg')
            #print 'Subimg', subimg_index, 'from img - [', i, ':', i+subimg_height, ',', j, ':', j+subimg_width, ']'
            #cv2.imshow('Subimg', subimg)
            #cv2.waitKey(0)

            for ii in np.arange(0, subimg.shape[0], block_height):
                for jj in np.arange(0, subimg.shape[1], block_width):
                    block = subimg[ii:ii + block_height, jj:jj + block_width]
                    #print ii, ii + block_height
                    #print jj, jj + block_width
                    # DEBUG
                    #cv2.namedWindow('Block')
                    #print '\tBlock', ii_begin/common_block_height*33+jj_begin/common_block_width, '- size', block.shape[0], 'x', block.shape[1]
                    #cv2.imshow('Block', block)

                    subblock1_mean = np.mean(block[0:block.shape[0] / 2, 0:block.shape[1] / 2])
                    subblock2_mean = np.mean(block[0:block.shape[0] / 2, block.shape[1] / 2:block.shape[1]])
                    subblock3_mean = np.mean(block[block.shape[0] / 2:block.shape[0], 0:block.shape[1] / 2])
                    subblock4_mean = np.mean(
                        block[block.shape[0] / 2:block.shape[0], block.shape[1] / 2:block.shape[1]])
                    subblocks_means = np.array([subblock1_mean, subblock2_mean, subblock3_mean, subblock4_mean])

                    m_values = np.abs(subblocks_means.dot(DIRECTION_FILTERS))
                    direction_index = m_values.argmax(axis=0)



                    if m_values[direction_index] > DIRECTION_THRES:
                        #print subimg_index, direction_index
                        hists[subimg_index, direction_index] += 1

                    # DEBUG
                    # print '\t\tblock means:', subblocks_means
                    # print '\t\tm values:', m_values
                    # print '\t\thistogram', hists[subimg_index, :]
                    # cv2.waitKey(0)

                    block_index += 1

            subimg_index += 1

    hists = hists / blocks_per_subimg

    quant_hist = np.zeros([16 * 5], dtype=np.uint8)
    for i in range(16):
        quant_hist[(i * 5):(i * 5 + 5)] += np.abs(np.ones([8, 1]) * hists[i, :] - QUANTIZER_MATRIX).argmin(axis=0).astype(np.uint8)

    global_hist = np.sum(hists, axis=0) / 16
    quant_global_hist = np.abs(np.ones([8, 1]) * global_hist - QUANTIZER_MATRIX).argmin(axis=0)

    semiglobal_hist = np.array([(hists[0, :] + hists[4, :] + hists[8, :] + hists[12, :]) / 4,
                                (hists[1, :] + hists[5, :] + hists[9, :] + hists[13, :]) / 4,
                                (hists[2, :] + hists[6, :] + hists[10, :] + hists[14, :]) / 4,
                                (hists[3, :] + hists[7, :] + hists[9, :] + hists[15, :]) / 4,
                                (hists[0, :] + hists[1, :] + hists[2, :] + hists[3, :]) / 4,
                                (hists[4, :] + hists[5, :] + hists[6, :] + hists[7, :]) / 4,
                                (hists[8, :] + hists[9, :] + hists[10, :] + hists[11, :]) / 4,
                                (hists[12, :] + hists[13, :] + hists[14, :] + hists[15, :]) / 4,
                                (hists[0, :] + hists[1, :] + hists[4, :] + hists[5, :]) / 4,
                                (hists[2, :] + hists[3, :] + hists[6, :] + hists[7, :]) / 4,
                                (hists[8, :] + hists[9, :] + hists[12, :] + hists[13, :]) / 4,
                                (hists[10, :] + hists[11, :] + hists[14, :] + hists[15, :]) / 4,
                                (hists[5, :] + hists[6, :] + hists[9, :] + hists[10, :]) / 4])

    quant_semiglobal_hist = np.zeros([13 * 5], dtype=np.uint8)
    for i in range(13):
        quant_semiglobal_hist[(i * 5):(i * 5 + 5)] += np.abs(
            np.ones([8, 1]) * semiglobal_hist[i, :] - QUANTIZER_MATRIX).argmin(axis=0).astype(np.uint8)

    # DEBUG
    #print 'Local histograms'
    #print hists
    #print 'Quantified histogram'
    #print quant_hist.reshape([16, 5])
    #print 'Global histogram'
    #print global_hist
    #print 'Quantified global histogram'
    #print quant_global_hist
    #print 'Semiglobal histograms'
    #print semiglobal_hist
    #print 'Quantified semiglobal histogram'
    #print quant_semiglobal_hist
    #print 'Number of subimgs -', subimg_index
    #print 'Number of blocks -', block_index
    #cv2.imshow('img', img)
    return hists, quant_hist, global_hist, quant_global_hist, semiglobal_hist, quant_semiglobal_hist