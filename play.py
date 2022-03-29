"""
In this example, we demonstrate how to create simple camera viewer using Opencv3 and PyQt5
Author: Berrouba.A
Last edited: 21 Feb 2018
"""

# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer


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

data = 0;

from main import *

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.logic=0
        self.value=1

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.pushButton.clicked.connect(self.controlTimer)
        self.ui.pushButton_2.clicked.connect(self.capture)
        self.ui.pushButton_3.clicked.connect(self.platedetection)


    def capture(self):
        self.logic=2
    
    def platedetection(self):
        img = cv2.imread('/home/pi/LVQ/image/1.png',0)
        # convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = img.shape
        step = channel * width
       # create QImage from image
        qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_4.setPixmap(QPixmap.fromImage(qImg))
                
        blnKNNTrainingSuccessful = Deteksi_Karakter.loadKNNDataAndTrainKNN()  

        if not blnKNNTrainingSuccessful:  
            print("\nerror: Pelatihan tidak berhasil\n")  # Pesan error
            return  # Program akan otomatis berhenti
    # end if

        imgOriginalS: None = cv2.imread("/home/pi/LVQ/image/1.png")  # Proses membaca Citra dari folder Sample_Plat

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

        #cv2.imshow("Citra Plat Original", imgOriginalScene)  # Menampilkan Citra Original

        if len(listOfPossiblePlates) == 0:  # Kondisi ketika plat tidak terdeteksi
            print("\nTidak ada nomor plat yang terdeteksi\n")  # Menampilkan Pesan Error
        else:
        # Kondisi ketika plat miring ataupun masih terdeteksi oleh program
        # Mengurutkan daftar kemungkinan nomor plat dengan metode DESCENDING ( diurutkan dari jumlah karakter terbanyak ke jumlah karakter yang paling sedikit)
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # Misalkan plat dengan karakter yang dapat dikenali (plat diurutkan berdasarkan urutan descending) adalah plat yang sebenarnya
            licPlate = listOfPossiblePlates[0]

        # Menampilkan Citra yang telah di Crop dan Citra Threshold
            #cv2.imshow("Citra Plat", licPlate.imgPlate)
            #cv2.imshow("Citra Threshold", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:  # Kondisi ketika karakter tidak dikenali
                print("\nTidak ada karakter yang terdeteksi\n\n")  # Pesan error
                self.ui.label_2.setText("Tidak Ada Plat Nomor")
                return  # Program akan berhenti
        # end if

        # Membuat rectangle di sekitar plat dengan karakter yang di kenali
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        # Menampilkan Nomor Plat kedalam print out program
            print("\nNomor Plat dari citra yang di deteksi yaitu = " + licPlate.strChars + "\n")
            print("----------------------------------------")
            self.ui.label_2.setText(licPlate.strChars)

        # Membuat nomor plat yang terdeteksi ke dalam Citra hasil
            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        # Menampilkan Citra hasil dengan nomor plat yang terdeteksi
            #cv2.imshow("Citra Plat Hasil", imgOriginalScene)

        # Menyimpan Citra Hasil kedalam file berekstensi.jpeg
            #cv2.imwrite("Citra_Hasil/citra_hasil_30.jpeg", imgOriginalScene)

    # end if else

        cv2.waitKey(0)

        return


# end main
        
    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = images.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(images.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label_3.setPixmap(QPixmap.fromImage(qImg))
                
        if(self.logic==2):
            self.value=self.value+1
            cv2.imwrite('/home/pi/LVQ/image/1.png', image)
                    
            self.logic=1
        

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.pushButton.setText("Stop Camera")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.pushButton.setText("Show Camera")

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # Membuat Rectangle

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED,
             5)  # Membuat 4 garis Hijau yang membentuk Rectangle
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 5)
# end function


###################################################################################################
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


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
    
    


