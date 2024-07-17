import cv2
import cv2 as cv
import numpy as np
import math
import sys

############ project files import ####################
from tracker import filter
#

STATIC_NONE_TRACK_ZONE_X = 170
STATIC_NONE_TRACK_ZONE_OTHER = 10
STATIC_FRAME_SKIP = 30

############### For 1920x1080    #####################
STATIC_FRAME_RESIZE_X = 3
STATIC_FRAME_RESIZE_Y = 4
STATIC_FRAME_RESIZE_BBOX = 4

"""
Пример:
    Для интеграции куда либо используются следующие функции
    
    import nanoTracking.py                  ### подключение класса
    
    ...
    
    track = nanoTracking.nanoTracking()     ### объект
    
    ...
    
    track.newPoint(x, y, hR, wR)            ### Старт трекинга, объвление участка трекинга (x,y,h,w) cv2.Rect: верхний левый угол области - x,y. h,w - ширина и высота
    
    ...
        while(...):
            ...
            
            ret2, frame2, point2, rect = track.getTrack(frame)     ### функции отслеживания области return: ret = True/False, frame2 = np.array(), point2 = [x,y], rect = [h, w]
            
            ...
    
    track.trackOff()                                                ### Отключение трекинга
    
"""



def _rect_to_sr(rect):
    """
     Перевод из формата cv2.Rect в формат FincoPlayer где x,y - центр, h,w - ширина и высота
    :param rect: cv2.Rect
    :return np.int32[4], где pt[0], pt[1] - центр, pt[2], pt[3] - ширина и высота
    """
    pt = np.zeros(4, np.int32)
    pt[0] = rect[0]
    pt[1] = rect[1]
    pt[2] = rect[2]
    pt[3] = rect[3]
    return pt

class nanoTracking():
    """
           Класс трекинга кастомного участка
    """
    def __init__(self):
        """
         Инициализация класса
         :param delta: debug param
         :param point2T: маcсив np.int32[2], верхняя левая точка прямоугольника
         :param hw:  маcсив np.int32[2], ширина и высота прямоугольника
         :param trackIsOn: bool, True - трекинг включен, False - выключен
         :param initF: bool, True - нейронка инциализирована, False - неинициализирована
         :param tracker: виртуал  класс cv2.Tracker осуществляющий обработку изображения
         :param filter: класс фильтра стабилизатора
         :param frameChecker: счетчик кадров на которых ожидается появлении потерянной области, если STATIC_FRAME_SKIP == frameChecker => trackOff()
         в случае успешного нахождения области на кадре frameChecker = 0
        """
        ############## Surface point #####################
        self.delta = np.zeros(2, np.int32)
        self.point2T = np.zeros(2, np.int32)
        self.hw = np.zeros(2, np.int32)
        self.hw[0] = 30
        self.hw[1] = 50

        ############## Flag track and init NN #############
        self.trackIsOn = False
        self.initF = False

        ############## Init NN ############################
        params = cv.TrackerNano.Params()
        params.backbone = 'models/nanotrack_backbone_sim.onnx'
        params.neckhead = 'models/nanotrack_head_sim.onnx'
        self.tracker = cv.TrackerNano.create(params)

        ############## Castom stabilizer ###################
        self.filter = filter.stabilizer()
        self.filter.sethw(self.hw)

        ############### Unit test for cProfiler #############################
        #self._resourceTest()

        ########## Checker frame when lost object ##########
        self.frameChecker = 0

    def _resourceTest(self):
        """
         Функция юнит теста для видео test.wmv
        :return
        """
        self.point2T[0] = 100
        self.point2T[1] = 100
        self.delta[0] = 0
        self.delta[1] = 0
        self.initF = False
        self.trackIsOn = True

    def setTrackingSurface(self):
        """
         Функция инициализации виртуального класса opencv TrackerNano, установка участка отслеживания
        :return
        """
        bbox = (int(self.point2T[0]-int(self.hw[0]/2)), int(self.point2T[1]-int(self.hw[1]/2)), int(self.hw[0]), int(self.hw[1]))
        self.tracker.init(self.frame, bbox)
        self.tracker.update(self.frame)

    def newPoint(self, x, y, hR, wR):
        """
         Установка участка для отслеживания, очистка данных
        :param x: верхний левый угол участка по Ox
        :param y: верхний левый угол участка по Oy
        :param hR: ширина участка
        :param wR: высота участка
        :return None
        """
        self.trackIsOn = True
        self.initF = False
        self.point2T[0] = int(x/STATIC_FRAME_RESIZE_X)
        self.point2T[1] = int(y/STATIC_FRAME_RESIZE_Y)
        self.hw[0] = int(wR/STATIC_FRAME_RESIZE_BBOX)
        self.hw[1] = int(hR/STATIC_FRAME_RESIZE_BBOX)
        self.delta[0] = 0
        self.delta[1] = 0
        self.scale = 1
        self.filter.clr()
        self.frameChecker = 0

    def trackOff(self):
        """
         Отключение трекинга участка, очистка  данных
        :return None
        """
        self.trackIsOn = False
        self.initF = False
        self.delta[0] = 0
        self.delta[1] = 0
        self.hRect = 0
        self.wRect = 0
        self.scale = 1
        self.frameChecker = 0

    def getTrack(self, frame):
        """
         Обработка кадра при включенном трекере (trackIsOn = True), при initF = False осуществляется инициация трекинга функция setTrackingSurface
        фильтрация нормализация бокса осуществляется классом filter, edgesFrame - ограничивает трекинг в зонах риска для нынешней версии аппаратной платформы
        :param frame: кадр, уже ресайзнутый.(1920x1080 = 640х270 = frame) Уменьшение в 4 раза по Оx, в 3 раза по Oy ( временное решение )
        :return bool, image, np.int32[2], np.int32[4]
        """
        if frame is not None:
            h, w = frame.shape[:2]
            frame = cv.resize(frame, (int(w/STATIC_FRAME_RESIZE_X), int(h/STATIC_FRAME_RESIZE_Y)))
            self.frame = frame
            ok = False
            if self.trackIsOn == True:
                if self.initF == False:
                    self.setTrackingSurface()
                    self.initF = True
                ok, bbox = self.tracker.update(frame)
                if ok:
                    self.frameChecker = 0
                    self.point2T[0] = bbox[0]
                    self.point2T[1] = bbox[1]
                    self.hw[0] = bbox[2]
                    self.hw[1] = bbox[3]
                    print("[SCORE]" + str(self.tracker.getTrackingScore()))
                    res, self.point2T = self.filter.predict(self.point2T)
                    # ok = self.edgesFrame()
                else:
                    self.frameChecker += 1
                    if self.frameChecker == STATIC_FRAME_SKIP:
                        self.trackOff()
                        return False, frame, _rect_to_sr(self.resizeResults())
                    else:
                        return True, self.frame, _rect_to_sr(self.resizeResults())
            return ok, self.frame, _rect_to_sr(self.resizeResults())
        else:
            return False, None, None

    def resizeResults(self):
        """
        :return np.int32[4] x*scale, y*scale, h*scale, w*scale
        """
        pt = np.zeros(4, np.int32)
        pt[0] = self.point2T[0]*STATIC_FRAME_RESIZE_X
        pt[1] = self.point2T[1]*STATIC_FRAME_RESIZE_Y
        pt[2] = self.hw[0]*STATIC_FRAME_RESIZE_BBOX
        pt[3] = self.hw[1]*STATIC_FRAME_RESIZE_BBOX
        return pt

    def edgesFrame(self):
        """
         Ограничения трекинга в зонах возникновения некоректных срабатываний
        :return None
        """
        h, w = self.frame.shape[:2]
        if self.point2T[0] < self.hw[0]+STATIC_NONE_TRACK_ZONE_X/STATIC_FRAME_RESIZE_X or self.point2T[1] < self.hw[1]+STATIC_NONE_TRACK_ZONE_OTHER or self.point2T[0] > w-self.hw[0]-STATIC_NONE_TRACK_ZONE_OTHER or self.point2T[1] > h - self.hw[1]-STATIC_NONE_TRACK_ZONE_OTHER:
            self.trackOff()
            return False
        return True

    #def auto_bbox_creator(self, frame, bbox):
