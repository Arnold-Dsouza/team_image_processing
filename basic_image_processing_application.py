import cv2
import numpy as np


def displayImage(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def convrtToGrayScale(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray Image', grayImage)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def binaryImage(img):
    ret, thresh = cv2.threshold(cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), 127, 255, 0)
    cv2.imshow('binary Image', thresh)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def convertToHSV(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv image', hsvImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def resizeImage(img):
    RESIZE_WIDTH = 300
    RESIZE_HEIGHT = 300
    imgResize = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT),
                           interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Resized Image', imgResize)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def rotateImage(img):
    rotateImg = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Rotated Image', rotateImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def cropImage(img):
    # [start row: end row, start column: end column]
    cropImg = img[0:200, 0:200]
    cv2.imshow('Cropped Image', cropImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def blurImage(img):
    blurImg = cv2.blur(img, (4, 4))
    cv2.imshow('Blurred Image', blurImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def sharpenImage(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpImg = cv2.filter2D(img, -1, kernel)
    cv2.imshow('Sharpened Image', sharpImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def edgeDetection(img):
    edgeImg = cv2.Canny(img, 100, 200)
    cv2.imshow('Edge Image', edgeImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


img = cv2.imread('cat.jpg', 1)
h, w, c = img.shape
print('Height and Width before resize: ', h, w)

displayImage(img, 'image')
status = cv2.imwrite('cat_copy.jpg', img)
print("status", status)
convrtToGrayScale(img)
binaryImage(img)
convertToHSV(img)
resizeImage(img)
rotateImage(img)
cropImage(img)
blurImage(img)
sharpenImage(img)
