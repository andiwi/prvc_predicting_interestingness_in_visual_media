import cv2
from readImgs import readImgs

def drawRuleOfThirdLines():
    print("Hello")

    directory = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\ruleofthirds\\part2'

    imgs = readImgs(directory)
    imgs = drawGrid(imgs)

    # save imgs
    for imgName, img in imgs.iteritems():
        if not os.path.exists(directory + '\\modified\\'):
            os.makedirs(directory + '\\modified\\')

        imgPath = directory + '\\modified\\' + imgName
        success = cv2.imwrite(imgPath, img)

    print("Finished")

def drawGrid(imgs):

    for img in imgs.itervalues():
        height, width, channels = img.shape

        cv2.line(img, (0, height/3), (width, height/3), (255, 0, 0), 1)
        cv2.line(img, (0, (height/3)*2), (width, (height/3)*2), (255, 0, 0), 1)

        cv2.line(img, (width/3, 0), (width/3, height), (255, 0, 0), 1)
        cv2.line(img, ((width / 3)*2, 0), ((width / 3)*2, height), (255, 0, 0), 1)

    return imgs


