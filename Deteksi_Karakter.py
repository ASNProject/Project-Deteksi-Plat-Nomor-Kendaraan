# DetectChars.py
import os

import cv2
import numpy as np
import math
import random
import Main_Deteksi_Plate
import Preprocessing_Citra
import Possible_Karakter


kNearest = cv2.ml.KNearest_create()

# Konstanta untuk checkIfPossibleChar, ini digunkan untuk menidentifikasi satu Possible (Kemungkinan) Karakter
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 90

# Konstanta untuk membandingkar 2 kemungkinan karakter
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# Konstanta lainnya
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # deklarasi empty lists,
    validContoursWithData = []              # digunakan untuk mengisi list data

    try:
        npaClassifications = np.loadtxt("Classifications.txt", np.float32)                  # Membaca data training classifications.txt dari proses pengenalan karakter
    except:
        print("error, Gagal load file classifications.txt\n")                               # Pesan error
        os.system("pause")
        return False                                                                        # Akan mengembalikan nilai 0
    # end try

    try:
        npaFlattenedImages = np.loadtxt("Flattened_Images.txt", np.float32)                 # Membaca data training Citra dari proses pengenalan karakter
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")              # pesan error
        os.system("pause")
        return False                                                                        # Akan mengembalikan nilai 0
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))            # reshape ke numpy array 1d

    kNearest.setDefaultK(1)                                                                  # setting default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)                # pelatihan KNN

    return True                                                                              # jika proses pelatihan berhasil akan mengembalikan nilai true
# end function

def detectCharsInPlates(listOfPossiblePlates):
    global height, width
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # Kondisi ketika kemungkinan plat yang terdeteksi adalah null
        return listOfPossiblePlates             # maka akan mengembalikan listOfPossiblePlates
    # end if

    # Pada proses dibawah ini akan memastikan kemungkinan plat yang dapat terdeteksi setidaknya satu plat

    for possiblePlate in listOfPossiblePlates:          # looping untuk setiap kemungkinan plat

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocessing_Citra.preprocess(possiblePlate.imgPlate)     # preprocessing untuk grayscale dan threshold

        if Main_Deteksi_Plate.showSteps:
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if

        # resize plat untuk memudahkan proses threshold
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

        # threshold lagi untuk menghilangkan area gray
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main_Deteksi_Plate.showSteps:
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if #

        # mencari semua kemungkinan karakter yang terdapat pada plat
        # fungsi ini pertama-tama digunakan untuk menemukan semua kontur, tetapi hanya mencakup kontur yang mungkin bisa menjadi karakter (belum dibandingkan dengan karakter lain)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main_Deteksi_Plate.showSteps:
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # hapus list contour

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main_Deteksi_Plate.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if

        # mengidentifikasi daftar semua karakter yang mungkin, lalu menemukan karakter yang cocok di dalam plat
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main_Deteksi_Plate.showSteps:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if

        if len(listOfListsOfMatchingCharsInPlate) == 0:			# Kondisi ketika tidak ada karakter yang cocok dan ditemukan pada plat

            if Main_Deteksi_Plate.showSteps:
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if

            possiblePlate.strChars = ""
            continue						# kembali ke awal lagi dari for loop
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                                                      # mencocokan setiap kemungkinan daftar karakter yang cocok
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)               # mengurutkan karakter yang telah terdeteksi dari kiri kekanan
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])    # dan menghapus karakter yang tumpang tindih (dan tidak jelas)
        # end for

        if Main_Deteksi_Plate.showSteps:
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if #

        # dari semua kemungkinan plat, anggaplah plat yang memiliki karakter panjang itu yang sebenarnya
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # loop melalui semua vektor karakter yang cocok, dapatkan indeks yang memiliki karakter paling banyak
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # asumsikan bahwa daftar karakter yang paling panjang di dalam plat adalah daftar karakter yang sebenarnya
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main_Deteksi_Plate.showSteps:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main_Deteksi_Plate.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main_Deteksi_Plate.showSteps:
            print("Karakter ditemukan dalam plat " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if

    # end of
    # akhir dari looping yang menggunakan beberapa fungsu

    if Main_Deteksi_Plate.showSteps:
        print("\nPendeteksian Karakter Selesai, Klik image atau press a key untuk continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # mengembalikan nilai dari value
    contours = []
    imgThreshCopy = imgThresh.copy()


    # Jika menggunakan OpenCv 3 -> # imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Mencari semua kontur pada plat
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:                               # for each contour
        possibleChar = Possible_Karakter.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # Jika pada kontur terdapat kemungkinan karakter, perhatikan jangan bandingkan dengan karakter lain
            listOfPossibleChars.append(possibleChar)       # Maka tambahkan ke List kemungkinan karakter
        # end if
    # end if

    return listOfPossibleChars
# end function


def checkIfPossibleChar(possibleChar):
    # fungsi ini adalah 'operan pertama' yang melakukan pengecekan kasar pada kontur untuk melihat apakah itu bisa menjadi char, perhatikan bahwa kita belum membandingkan karakter dengan karakter lain
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

def findListOfListsOfMatchingChars(listOfPossibleChars):
    # dengan fungsi ini, kita mulai dengan semua karakter yang mungkin ada dalam satu list besar
    # Tujuan dari fungsi ini adalah untuk mengatur kembali satu karakter list besar ke dalam karakter list-list yang cocok,
    # perhatikan bahwa karakter yang tidak ditemukan dalam plat maka pencocokan tidak perlu dipertimbangkan lebih lanjut
    listOfListsOfMatchingChars = []                  # mengembalikan nilai value

    for possibleChar in listOfPossibleChars:                        # for each possible char pada satu list besar dari karakter
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # mencari semua karakter pada list besar dan mencocokan pada karakter sebelumnya

        listOfMatchingChars.append(possibleChar)                # tambahkan juga karakter saat ini ke list kemungkinan karakter yang cocok saat ini

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # kondisi jika list karakter yang sesuai saat ini tidak cukup panjang untuk membentuk kemungkinan plat
            continue                                                    # kembali ke atas untuk for loop dan coba lagi dengan char berikutnya, perhatikan bahwa itu tidak perlu
                                                                        # untuk menyimpan list dengan cara apa pun karena
                                                                        # tidak memiliki karakter yang cukup untuk dijadikan kemungkinan plat
        # end if

        # jika sampai proses di sini, list saat ini lolos uji sebagai "grup" atau "gugus" karakter yang cocok
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # jadi tambahkan ke dalam daftar karakter yang cocok

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # hapus list karakter yang cocok saat ini dari list besar tidak digunakan karakter yang sama dua kali,
        # pastikan untuk membuat list besar baru
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # untuk setiap list karakter yang cocok ditemukan oleh panggilan rekursif
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # tambahkan ke list asli untuk list karakter yang cocok
        # end for

        break

    # end for

    return listOfListsOfMatchingChars
# end function


def findListOfMatchingChars(possibleChar, listOfChars):
    # Tujuan dari fungsi ini adalah untuk menyimpan kemungkinan karakter dan lists
    # Temukan semua karakter dalam z besar yang cocok dengan karakter tunggal yang mungkin, dan kembalikan karakter yang cocok sebagai daftar
    listOfMatchingChars = []                # this will be the return value

    for possibleMatchingChar in listOfChars:        # for each karakter in big list
        if possibleMatchingChar == possibleChar:

            continue
        # end if

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        # check jika karakter cocok
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # jika karakter cocok tambahkan ke list kecocokan karakter
        # end if
    # end for

    return listOfMatchingChars                  # return hasil dari kecocokan karakter
# end function

# gunakan theorema pythagoras untuk menghitung jarak 2 karakter
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

# gunakan trigonometri dasar untuk menghitung sudut antar karakter
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                                   # periksa untuk memastikan bahwa tidak membagi dengan nol jika posisi pusat X sama, pembagian float dengan nol akan menyebabkan crash pada Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # hitung sudut
    else:
        fltAngleInRad = 1.5708
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)   # hitung sudut dalam derajat

    return fltAngleInDeg
# end function

# Jika memiliki dua karakter yang tumpang tindih atau untuk menutup satu sama lain untuk menjadi karakter yang terpisah, hapus karakter bagian dalam (lebih kecil),
# ini untuk mencegah memasukkan karakter yang sama dua kali jika dua kontur ditemukan untuk karakter yang sama,
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # mengembalikan nilai value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:                                    # jika kondisi karakter saat ini tidak sama dengan sebelumnya

                # jika karakter saat ini dan lainnya memiliki titik tengah dan hampir sama
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # Jika pada proses ini maka ditemukan karakter yang tumpang tindih
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # kondisi karakter saat ini lebih kecil dari yang lain
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # jika karakter saat ini belum dihapus pada kondisi sebelumnya
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # menghapus karakter sebelumnya
                        # end if
                    else:                                                                       # jika karakter lain lebih kecil dari karakter saat ini
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function


# fungsi pengenalan karakter
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               # mengembalikan value

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # mengurutkan karakter dari kiri ke kanan

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                         # gunakan warna threshold dan gray agar dapat membentuk kontur

    for currentChar in listOfMatchingChars:                                             # for each karakter pada plat
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main_Deteksi_Plate.SCALAR_GREEN, 2)     # membuat green box pada sekitar karakter

        # crop gambar dari threshold
        imgROI = imgThresh[currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))               # resize gambar, diperlukan untuk pengenalan karakter

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # flatten image pada numpy array 1d

        npaROIResized = np.float32(npaROIResized)                                                               # konversi dari 1d numpy array dari integer ke 1d numpy array dari floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)                      # panggil kembali fungsi findNearest !!!

        strCurrentChar = str(chr(int(npaResults[0][0])))                                                        # dapatkan karakter dari hasil

        strChars = strChars + strCurrentChar                                                                    # gabungkan semua karakter

    # end for

    if Main_Deteksi_Plate.showSteps:
        cv2.imshow("10", imgThreshColor)
    # end if

    return strChars
# end function
