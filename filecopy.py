import shutil
import csv
import os

def filecopy():
    print("Start Copying Interesting Images")
    csvPath = 'D:\PR aus Visual Computing\Interestingness16data\groundtruth.csv'
    imgsPath = 'D:\PR aus Visual Computing\Interestingness16data\\allvideos'

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

    '''
    #copy interesting files
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
    '''


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



    '''
    # copy uninteresting files
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
    '''

    #copy interesting files in single directory
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