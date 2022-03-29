# Preprocessing_Citra.py

import cv2
import numpy as np

# Hasil

# berhasil pada plat 1.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -2

# berhasil pada plat 2.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -3

# Berhasil pada plat 3.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.2 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 39
# ADAPTIVE_THRESH_WEIGHT = -16

# Berhasil pada plat 4.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = 0

# Berhasil pada plat 5.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.2 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 27
# ADAPTIVE_THRESH_WEIGHT = -6

# Berhasil pada plat 6.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -1

# Berhasil pada plat 7.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = 2

# Berhasil pada plat 8.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 21
# ADAPTIVE_THRESH_WEIGHT = -2

# Berhasil pada plat 9.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.3 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -15

# Berhasil pada plat 10.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 27
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 11.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 27
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 12.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 13.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 21
# ADAPTIVE_THRESH_WEIGHT = 1.2

# Berhasil pada plat 14.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.2 dan PLATE_HEIGHT_PADDING_FACTOR = 1.0
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 41
# ADAPTIVE_THRESH_WEIGHT = -6

# Berhasil pada plat 15.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.4 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (7, 7)
# ADAPTIVE_THRESH_BLOCK_SIZE = 25
# ADAPTIVE_THRESH_WEIGHT = -16

# Berhasil pada plat 16.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.4 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (7, 7)
# ADAPTIVE_THRESH_BLOCK_SIZE = 25
# ADAPTIVE_THRESH_WEIGHT = -16

# Berhasil pada plat 17.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -25

# Berhasil pada plat 18.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -17

# Berhasil pada plat 19.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -5

# Berhasil pada plat 20.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -9

# Berhasil pada plat 21.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -9

# Berhasil pada plat 22.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -9

# Berhasil pada plat 23.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = 2

# Berhasil pada plat 24.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.7 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = -4

# Berhasil pada plat 25.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 21
# ADAPTIVE_THRESH_WEIGHT = -7

# Berhasil pada plat 26.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 27.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 28.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 29.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = -18

# Berhasil pada plat 30.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 25
# ADAPTIVE_THRESH_WEIGHT = -4

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 25
ADAPTIVE_THRESH_WEIGHT = -4

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 5)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    # kernel = np.ones((5, 5), np.uint8)
    # imgThresh = cv2.dilate(imgThresh, kernel, iterations=0)
    # imgThresh = cv2.erode(imgThresh, kernel, iterations=0)

    return imgGrayscale, imgThresh
# end function

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 1), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function
