# DetectPlates.py

import cv2
import numpy as np
import math
import Main_Deteksi_Plate
import random

import Preprocessing_Citra
import Deteksi_Karakter
import Possible_Plat
import Possible_Karakter

# Inisiasi padding pada Cropping Plat nomor
PLATE_WIDTH_PADDING_FACTOR = 1.7
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # Return dari value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main_Deteksi_Plate.showSteps:  # show steps
        cv2.imshow("0", imgOriginalScene)
    # end if

    imgGrayscaleScene, imgThreshScene = Preprocessing_Citra.preprocess(imgOriginalScene)         # preprocess untuk memperoleh grayscale dan threshold

    if Main_Deteksi_Plate.showSteps:
        cv2.imshow("Konversi Ke GrayScale", imgGrayscaleScene)
        cv2.imshow("Threshold", imgThreshScene)
    # end if

    # mencari semua kemungkinan karakter pada plat
    # fungsi ini pertama-tama menemukan semua kontur, kemudian hanya menyertakan kontur yang bisa menjadi karakter (belum dibandingkan dengan karakter lain)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main_Deteksi_Plate.showSteps:
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))

        imgContours = np.zeros((height, width, 1), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main_Deteksi_Plate.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if

    # diberikan daftar semua karakter yang mungkin, temukan kelompok karakter yang cocok
    # pada langkah-langkah selanjutnya setiap kelompok karakter yang cocok akan dikenali sebagai plat
    listOfListsOfMatchingCharsInScene = Deteksi_Karakter.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main_Deteksi_Plate.showSteps:
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each untuk mengelompokkan kecocokan karakter
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         # ektraksi plat

        if possiblePlate.imgPlate is not None:                          # jika plat ditemukan
            listOfPossiblePlates.append(possiblePlate)                  # tambahkan pada list kemungkinan plat
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")

    if Main_Deteksi_Plate.showSteps:
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main_Deteksi_Plate.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main_Deteksi_Plate.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main_Deteksi_Plate.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main_Deteksi_Plate.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("Kemungkinan Plat " + str(i) + ", klik image dan press a key untuk continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\nPendeteksian plat selesai, klik image dan press a key untuk memulai pengenalan karakter . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # mengembalikan value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    # mencari semua kontur
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST,
                                                           cv2.CHAIN_APPROX_SIMPLE)
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 1), np.uint8)

    for i in range(0, len(contours)):                       # for each kontur

        if Main_Deteksi_Plate.showSteps:
            cv2.drawContours(imgContours, contours, i, Main_Deteksi_Plate.SCALAR_WHITE)
        # end if

        possibleChar = Possible_Karakter.PossibleChar(contours[i])

        if Deteksi_Karakter.checkIfPossibleChar(possibleChar):              # jika kontur adalah karakter yang memungkinkan, jangan dibandingkan dengan karakter lain
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # menghitung kemungkinan banyaknya karakter
            listOfPossibleChars.append(possibleChar)                        # dan tambahkan pada list kemungkinan karakter
        # end if
    # end for

    if Main_Deteksi_Plate.showSteps:
        print("\nstep 2 - len(contours) = " + str(len(contours)))
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))
        cv2.imshow("2a", imgContours)
    # end if

    return listOfPossibleChars
# end function

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = Possible_Plat.PossiblePlate()                           # mengembalikan value

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # mengurutkan karakter dari kiri ke kanan

    # hitung titik tengah pada plat
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # hitung width dan height pada plat
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # hitung sudut koreksi pada plat
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = Deteksi_Karakter.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width dan height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped

    return possiblePlate
# end function
