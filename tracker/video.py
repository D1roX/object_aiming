import numpy as np
import cv2 as cv
import vidstab as vs


class video():
    def __init__(self, name="test.avi", stab=False):
        self.testpath = name
        self.cap = cv.VideoCapture(self.testpath)
        self.size = np.zeros(2, np.int32)
        self.stabilizer = vs.VidStab()
        self.stabF = stab
        self.frame = self.grab_frame()

    def grab_frame(self):
        if self.cap.isOpened() == True:
            ret, self.frame = self.cap.read()
            #self.frame = cv.resize(self.frame, [640, 480])
            if self.stabF == True:
                self.frame = self.stabilizer.stabilize_frame(input_frame=self.frame, smoothing_window=30)
            #self.frame = self.frame[0:480, 200:640]
            return self.frame
        else:
            print("cant open videostream")
            return None

    #def cropAndFiltred(self):

