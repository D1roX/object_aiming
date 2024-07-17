import cv2 as cv
import numpy as np
import math
import sys

PY3 = sys.version_info[0] == 3
if PY3:
    long = int

STATIC_NONE_TRACK_ZONE_X = 170
STATIC_NONE_TRACK_ZONE_OTHER = 10
STATIC_FRAME_SKIP = 30

############### For 1920x1080    #####################
STATIC_FRAME_RESIZE_X = 3
STATIC_FRAME_RESIZE_Y = 4
STATIC_FRAME_RESIZE_BBOX = 4

class trackingAffine():
    def __init__(self, filterFlag= False, detector_type='ORB', WindowController=True, winname="mainWin", ):
        ############## Image transformation marker: scale, delta, point #####################
        self.scalingRN = 1
        self.delta = np.zeros(2, np.int32)
        self.point2T = np.zeros(2, np.int32)

        ############## Mouse Event Controller using if UI from Opencv ######################
        self.trackIsOn = False

        ############# Other math var #######################################################
        self.prevAngle = 0
        self.localAngle = 0
        self.point2T_kallman = np.zeros(2, np.int32)
        ############# prev and actual descriptor ##########################################
        self.kpts1 = self.kpts2 = None
        self.desc1 = self.desc2 = None
        self.prevFrame = None
        self.frame = None
        ############ Drawing var ###########################################################
        self.wRect = 50
        self.hRect = 50
        self.th = 2
        self.color = (0, 0, 0)
        ################## Test ######################


        ############################# ORB or AKAZE or SIFT or SURF or BRIEF ################
        akaze = cv.AKAZE_create()
        sift = cv.SIFT_create()
        orb = cv.ORB_create()
        brisk = cv.BRISK_create()
        self.detector = None
        if detector_type == 'AKAZE':
            self.detector = akaze
        elif detector_type == 'SIFT':
            self.detector = sift
        elif detector_type == 'ORB':
            self.detector = orb
        elif detector_type == 'BRISK':
            self.detector = brisk
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        

        ##### test
        #self._resourceTest()
    def _resourceTest(self):
        self.point2T[0] = 100
        self.point2T[1] = 100
        self.delta[0] = 0
        self.delta[1] = 0
        self.scale = 1
        self.trackIsOn = True


    def newPoint(self, x, y, hR, wR):

        self.trackIsOn = True
        self.point2T[0] = x
        self.point2T[1] = y
        self.hRect = hR
        self.wRect = wR
        self.delta[0] = 0
        self.delta[1] = 0
        self.scale = 1
        self.prevFrame = None

    def trackOff(self):
        self.trackIsOn = False
        self.delta[0] = 0
        self.delta[1] = 0
        self.hRect = 0
        self.wRect = 0
        self.scale = 1


    def get_point_and_descriptors(self, img):
        """
         Получение кейпоинтов и их дескрипторов
        :param img: изображение с которого необходимо получить информацию
        :return: кейпоинты, дескрипторы
        """
        kpts, desc = self.detector.detectAndCompute(img, None)
        return kpts, desc

    def calculate_offset(self):
        """
         Поиск и сравнение совпадений фич двух кадров
        :return ret, inliners: матрица афиннго преобразования, вектор точек на основе которых производился рассчет из функции angleRotatedSearcher
        """
        matches = self.matcher.match(self.desc1, self.desc2)
        self.matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) == 0:
            return None, None
        elif len(matches) > 40:
            self.matches[:40]
        self.src_pts = src_pts = np.float32([self.kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.dst_pts = dst_pts = np.float32([self.kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return self.angleRotatedSearcher(src_pts, dst_pts)

    def angleRotatedSearcher(self, prev_pts, curr_pts, met=cv.RANSAC):
        """
         Поиск угла поворота между двумя кадрами
        :param prev_pts: предыдущие кейпоинты
        :param curr_pts: текущие кейпоинты
        :param met: метод, который используется для оценки аффинного преобразования между двумя наборами точек
        :return ret, inliners: матрица афиннго преобразования, вектор точек на основе которых производился рассчет
        """
        if len(prev_pts) != 0 and len(curr_pts) != 0:
            ret, inliners = cv.estimateAffinePartial2D(prev_pts, curr_pts, method=met, confidence=0.99)
        else:
            return None, None
        if ret is not None:
            angle = math.atan((-ret[0][1]) / ret[0][0])
            self.prevAngle = angle
            self.localAngle += angle  # * 57.2958
            self.scale = scale = ret[0][0] / math.cos(self.prevAngle)
            return ret, inliners
        else:
            return ret, inliners

    def doit(self, frame):
        """
         Фкн
        :param prev_pts: предыдущие кейпоинты
        :param curr_pts: текущие кейпоинты
        :param met: метод, который используется для оценки аффинного преобразования между двумя наборами точек
        :return ret, inliners: матрица афиннго преобразования, вектор точек на основе которых производился рассчет
        """
        frame = cv.blur(frame, (2, 2))
        if self.trackIsOn == True:
            if np.shape(self.prevFrame) == ():
                self.prevFrame = frame
                self.kpts1, self.desc1 = self.get_point_and_descriptors(self.prevFrame)
            else:
                self.frame = frame
                self.kpts2, self.desc2 = self.get_point_and_descriptors(self.frame)
                ret, inliners = self.calculate_offset()
                if ret is not None:
                    self.delta[0] = ret[0][2]
                    self.delta[1] = ret[1][2]
                    self.deltaCheck(ret)
                else:
                    self.trackIsOn = False
                    return self.trackIsOn, frame, (int(self.point2T[0]), int(self.point2T[1]), int(self.wRect), int(self.hRect))
                # if self.edgesFrame() == False:
                #     self.trackIsOn = False
                #     return self.trackIsOn, frame, (int(self.point2T[0]), int(self.point2T[1]), int(self.wRect), int(self.hRect))
                self.kpts1 = self.kpts2
                self.desc1 = self.desc2
                self.prevFrame = self.frame
            return self.trackIsOn, frame, (int(self.point2T[0]-self.wRect/2), int(self.point2T[1]-self.hRect/2), int(self.wRect), int(self.hRect))
        else:
            return self.trackIsOn, frame, None


    def deltaCheck(self, ret):
        x = self.point2T[0] * ret[0][0] + self.point2T[1] * ret[0][1] + self.scale*(ret[0][2]+0.5)       # Статическая погрешность
        y = self.point2T[0] * ret[1][0] + self.point2T[1] * ret[1][1] + self.scale*(ret[1][2]+0.3)       # Статическая погрешность
        self.point2T[0] = x
        self.point2T[1] = y

    def edgesFrame(self):
        h, w = self.frame.shape[:2]
        if self.point2T[0] < self.wRect+STATIC_NONE_TRACK_ZONE_X or self.point2T[1] < self.hRect+STATIC_NONE_TRACK_ZONE_OTHER or self.point2T[0] > w-self.wRect-STATIC_NONE_TRACK_ZONE_OTHER or self.point2T[1] > h - self.hRect-STATIC_NONE_TRACK_ZONE_OTHER:
            self.trackOff()
            return False
        return True
