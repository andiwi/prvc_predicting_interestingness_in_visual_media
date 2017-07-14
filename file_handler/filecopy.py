import shutil
import csv
import os

def filecopy():
    '''
    read groundtruth and copies images into interesting and uninteresting directories
    :return:
    '''
    print("Start Copying Interesting Images")
    csvPath = 'D:\PR aus Visual Computing\Interestingness17data\groundtruth.csv'
    imgsPath = 'D:\PR aus Visual Computing\Interestingness17data\\allvideos'

    # read interesting files from csv ground truth
    f = open(csvPath)
    csv_f = csv.reader(f)

    imgs_interesting = dict()
    imgs_uninteresting = dict()

    for row in csv_f:
        if row[2] == '1':
            if row[0] in imgs_interesting:
                imgs_interesting[row[0]].append(row[1])
            else:
                imgs_interesting[row[0]] = [row[1]]

        if row[2] == '0':
            if row[0] in imgs_uninteresting:
                imgs_uninteresting[row[0]].append(row[1])
            else:
                imgs_uninteresting[row[0]] = [row[1]]

    #copy interesting files in same directory
    for videoDir, imgs in imgs_interesting.iteritems():
        interestingDir = imgsPath + '\\' + videoDir + '\\images\\interesting'
        if not os.path.exists(interestingDir):
            os.makedirs(interestingDir)
    
        for img in imgs:
            interestingImgDirFrom = imgsPath + '\\' + videoDir + '\\images\\' + img
            interestingImgDirTo = imgsPath + '\\' + videoDir + '\\images\\interesting\\' + img
    
            print('Copying from: ' + interestingImgDirFrom + ' to: ' + interestingImgDirTo)
            shutil.copy2(interestingImgDirFrom, interestingImgDirTo)
    
    print('finished copying interesting files')


    # copy uninteresting files in same directory
    for videoDir, imgs in imgs_uninteresting.iteritems():
        uninterestingDir = imgsPath + '\\' + videoDir + '\\images\\uninteresting'
        if not os.path.exists(uninterestingDir):
            os.makedirs(uninterestingDir)

        for img in imgs:
            uninterestingImgDirFrom = imgsPath + '\\' + videoDir + '\\images\\' + img
            uninterestingImgDirTo = imgsPath + '\\' + videoDir + '\\images\\uninteresting\\' + img

            print('Copying from: ' + uninterestingImgDirFrom + ' to: ' + uninterestingImgDirTo)
            shutil.copy2(uninterestingImgDirFrom, uninterestingImgDirTo)

    print('finished copying uninteresting files')


    #copy interesting files in single directory
    for videoDir, imgs in imgs_interesting.iteritems():
        interestingDir = imgsPath + '\\images\\interesting'
        if not os.path.exists(interestingDir):
            os.makedirs(interestingDir)

        for img in imgs:
            interestingImgDirFrom = imgsPath + '\\' + videoDir + '\\images\\' + img
            interestingImgDirTo = imgsPath + '\\images\\interesting\\' + img

            print('Copying from: ' + interestingImgDirFrom + ' to: ' + interestingImgDirTo)
            shutil.copy2(interestingImgDirFrom, interestingImgDirTo)

    print('finished copying interesting files in single directory')

    #copy uninteresting files in single directory
    for videoDir, imgs in imgs_uninteresting.iteritems():
        uninterestingDir = imgsPath + '\\images\\uninteresting'
        if not os.path.exists(uninterestingDir):
            os.makedirs(uninterestingDir)

        for img in imgs:
            uninterestingImgDirFrom = imgsPath + '\\' + videoDir + '\\images\\' + img
            uninterestingImgDirTo = imgsPath + '\\images\\uninteresting\\' + img

            print('Copying from: ' + uninterestingImgDirFrom + ' to: ' + uninterestingImgDirTo)
            shutil.copy2(uninterestingImgDirFrom, uninterestingImgDirTo)

    print('finished copying uninteresting files in single directory')

def extract_interesting_uninteresting_images(dir):
    '''
    read groundtruth file and copies images into interesting and uninteresting directories
    :param: dir... the root directory of the dataset ex. './InterestingnessData16/devset' or './InterestingnessData16/testset
    :return:
    '''
    gt_path = os.path.join(dir, 'devset', 'annotations', 'devset-image.txt')

    dir_interesting = os.path.join(dir, 'devset', 'imgs_interesting')
    dir_uninteresting = os.path.join(dir, 'devset', 'imgs_uninteresting')

    #create directories
    if not os.path.exists(dir_interesting):
        os.makedirs(dir_interesting)

    if not os.path.exists(dir_uninteresting):
        os.makedirs(dir_uninteresting)

    #read ground truth file
    with open(gt_path) as f:
        gt_content = f.readlines()
        gt_content = [x.strip() for x in gt_content] #remove whitespace characters like `\n` at the end of each line
        gt_content = [x.split(',') for x in gt_content] #split by delimiter ','


    #copy images into interesting and uninteresting directory
    for row in gt_content:
        img_dir_from = os.path.join(dir, 'devset', 'videos', row[0], 'images', row[1])

        if row[2] == '1':
            #interesting
            shutil.copy2(img_dir_from, dir_interesting)

        elif row[2] == '0':
            #uninteresting
            shutil.copy2(img_dir_from, dir_uninteresting)