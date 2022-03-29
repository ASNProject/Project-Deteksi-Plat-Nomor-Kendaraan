import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    imgTrainingNumbers = cv2.imread("Citra_Plate_Training/training_0.jpg")

    # Resize Citra menjadi skala 80%
    scale_percent = 80
    width = int(imgTrainingNumbers.shape[1] * scale_percent / 100)
    height = int(imgTrainingNumbers.shape[0] * scale_percent / 100)
    dimensi = (width, height)
    # Citra Setelah di Resize
    imgTrainingResize = cv2.resize(imgTrainingNumbers, dimensi, interpolation=cv2.INTER_AREA)

    if imgTrainingNumbers is None:
        print("error: image not read from file \n\n")
        os.system("pause")
        return
    # end if

    imgGray = cv2.cvtColor(imgTrainingResize, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow("imgThresh", imgThresh)

    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTrainingResize,
                          (intX, intY),
                          (intX + intW, intY + intH),
                          (0, 179, 0),  # green
                          2)

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers.png", imgTrainingResize)

            intChar = cv2.waitKey(0)
            print(intChar)
            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print("\n\ntraining selesai !!\n")
    # print(fltClassifications)
    # print(npaFlattenedImages)
    np.savetxt("Classifications.txt", npaClassifications)
    np.savetxt("Flattened_Images.txt", npaFlattenedImages)
    # changeCaption()

    cv2.destroyAllWindows()  # remove windows from memory

    return


# def changeCaption():
#     data = np.loadtxt("classifications.txt")
#     i = 0
#     for a in data:
#         a = int(round(a))
#         if a == ord('a'):
#             data[i] = ord('A')
#         if a == ord('b'):
#             data[i] = ord('B')
#         if a == ord('c'):
#             data[i] = ord('C')
#         if a == ord('d'):
#             data[i] = ord('D')
#         if a == ord('e'):
#             data[i] = ord('E')
#         if a == ord('f'):
#             data[i] = ord('F')
#         if a == ord('g'):
#             data[i] = ord('G')
#         if a == ord('h'):
#             data[i] = ord('H')
#         if a == ord('i'):
#             data[i] = ord('I')
#         if a == ord('j'):
#             data[i] = ord('J')
#         if a == ord('k'):
#             data[i] = ord('K')
#         if a == ord('l'):
#             data[i] = ord('L')
#         if a == ord('m'):
#             data[i] = ord('M')
#         if a == ord('n'):
#             data[i] = ord('N')
#         if a == ord('o'):
#             data[i] = ord('O')
#         if a == ord('p'):
#             data[i] = ord('P')
#         if a == ord('q'):
#             data[i] = ord('Q')
#         if a == ord('r'):
#             data[i] = ord('R')
#         if a == ord('s'):
#             data[i] = ord('S')
#         if a == ord('t'):
#             data[i] = ord('T')
#         if a == ord('u'):
#             data[i] = ord('U')
#         if a == ord('v'):
#             data[i] = ord('V')
#         if a == ord('w'):
#             data[i] = ord('W')
#         if a == ord('x'):
#             data[i] = ord('X')
#         if a == ord('y'):
#             data[i] = ord('Y')
#         if a == ord('z'):
#             data[i] = ord('Z')
#         i = i + 1
#
#     # fltClassifications = np.array(intClassifications, np.float32)
#     hasil = np.array(data, np.float32)  # convert classifications list of ints to numpy array of floats
#     npaClassifications = hasil.reshape((hasil.size, 1))
#
#     np.savetxt("Classifications.txt", npaClassifications)


if __name__ == "__main__":
    main()
# end if
