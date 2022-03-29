# PossiblePlate.py
# Class untuk mendeteksi kemungkinan Plat yang terdapat pada Citra
class PossiblePlate:
    imgPlate: None
    # constructor
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None
        self.rrLocationOfPlateInScene = None
        self.strChars = ""
    # end constructor
# end class
