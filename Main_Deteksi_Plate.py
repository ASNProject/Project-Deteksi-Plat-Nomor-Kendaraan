# Main.py

import os
import cv2

import Deteksi_Karakter
import Deteksi_Plat

# Modul warna RGB
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class ContourWithData:
    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True


showSteps = False


def main():
    blnKNNTrainingSuccessful = Deteksi_Karakter.loadKNNDataAndTrainKNN()  # Proses percobaan pelatihan KNN

    if not blnKNNTrainingSuccessful:  # Jika pelatihan KNN Gagal
        print("\nerror: Pelatihan algoritma KNN tidak berhasil\n")  # Pesan error
        return  # Program akan otomatis berhenti
    # end if

    imgOriginalS: None = cv2.imread("Sample_Plat/1.png")  # Proses membaca Citra dari folder Sample_Plat

    # Resize Citra menjadi skala 80%
    scale_percent = 80
    width = int(imgOriginalS.shape[1] * scale_percent / 100)
    height = int(imgOriginalS.shape[0] * scale_percent / 100)
    dimensi = (width, height)
    # Citra Setelah di Resize
    imgOriginalScene = cv2.resize(imgOriginalS, dimensi, interpolation=cv2.INTER_AREA)

    if imgOriginalScene is None:  # Kondisi ketika Citra gagal di Load
        print("\nerror: citra tidak terbaca \n\n")  # Pesan error
        os.system("pause")  # Program akan jeda dan menampilkan pesan
        return  # Program Berhenti
    # end if

    listOfPossiblePlates = Deteksi_Plat.detectPlatesInScene(imgOriginalScene)  # Pendeteksian Plat dari Citra Original
    listOfPossiblePlates = Deteksi_Karakter.detectCharsInPlates(
        listOfPossiblePlates)  # Pendeteksian Karakter yang terdapat pada Plat Nomor

    cv2.imshow("Citra Plat Original", imgOriginalScene)  # Menampilkan Citra Original

    if len(listOfPossiblePlates) == 0:  # Kondisi ketika plat tidak terdeteksi
        print("\nTidak ada nomor plat yang terdeteksi\n")  # Menampilkan Pesan Error
    else:
        # Kondisi ketika plat miring ataupun masih terdeteksi oleh program
        # Mengurutkan daftar kemungkinan nomor plat dengan metode DESCENDING ( diurutkan dari jumlah karakter terbanyak ke jumlah karakter yang paling sedikit)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # Misalkan plat dengan karakter yang dapat dikenali (plat diurutkan berdasarkan urutan descending) adalah plat yang sebenarnya
        licPlate = listOfPossiblePlates[0]

        # Menampilkan Citra yang telah di Crop dan Citra Threshold
        cv2.imshow("Citra Plat", licPlate.imgPlate)
        cv2.imshow("Citra Threshold", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:  # Kondisi ketika karakter tidak dikenali
            print("\nTidak ada karakter yang terdeteksi\n\n")  # Pesan error
            return  # Program akan berhenti
        # end if

        # Membuat rectangle di sekitar plat dengan karakter yang di kenali
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        # Menampilkan Nomor Plat kedalam print out program
        print("\nNomor Plat dari citra yang di deteksi yaitu = " + licPlate.strChars + "\n")
        print("----------------------------------------")

        # Membuat nomor plat yang terdeteksi ke dalam Citra hasil
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        # Menampilkan Citra hasil dengan nomor plat yang terdeteksi
        cv2.imshow("Citra Plat Hasil", imgOriginalScene)

        # Menyimpan Citra Hasil kedalam file berekstensi.jpeg
        cv2.imwrite("Citra_Hasil/citra_hasil_30.jpeg", imgOriginalScene)

    # end if else

    cv2.waitKey(0)

    return


# end main


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # Membuat Rectangle

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED,
             5)  # Membuat 4 garis Hijau yang membentuk Rectangle
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 5)
# end function


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    # Titik buat area penulisan text pada Citra
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    # Titik bagian Kiri dari penulisan text pada Citra
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # Jenis font yang ditampilkan pada Citra
    fltFontScale = float(plateHeight) / 25.0
    intFontThickness = int(round(fltFontScale * 3.5))

    # Memanggil font dalam Citra dengan fungsi getTextSize pada OpenCv
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    # Lokasi horizontal text sama dengan plat
    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):  # Posisi ketika Plat berada di 3/4 dari Citra
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # Menulis karakter di bawah plat
    else:  # Posisi Plat ketika berada di 1/4 dari Citra
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # Menulis Karakter di atas plat
    # end if

    # unpack text size width dan height
    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    # Menulis Karakter text yang dikenali kedalam Citra
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_RED, intFontThickness)


# end function

if __name__ == "__main__":
    main()
