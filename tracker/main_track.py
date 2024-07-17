import math

import cv2 as cv
import numpy as np
from tracker import nanoTracking
from tracker import trackingAffine
"""
...
mt = main_track.mainTrack()
...
     mt.start_tracking_surf(x, y, 60, 60, 0) # Affine Tracking
                        OR
     mt.start_tracking_surf(x, y, 60, 60, 1) # NN Nano Tracking
    ...
    while(True):
        ...
        res, frame_res, rect = mt.videoThreading(frame) :return: bool (flag), np.array (frame), np.int32[4] (Rect-x,y,w,h)
        ...

"""

class mainTrack():
    """
     Класс осуществляющий обработку выбора трекинга, обработку задержки,  управляющий класс
     в планах добавить ватчдог
    """
    def __init__(self, param_dict=None):
        """
        :param param_dict: dict - параметры из файла конфигурации
        :param debug: bool - при True включается отрисовка окон  с рамками трекинга если он включен
        """
        if param_dict is not None:
            self.setParams(param_dict)
        else:
            self.initClassicParam()
        self.__reinit__()
        self.frame_list = list()
        self.type = 0

    def __reinit__(self):
        """
         Инициализация трекеров
        """
        self.track_affine = trackingAffine.trackingAffine(False, "ORB")
        self.track_nano = nanoTracking.nanoTracking()


    def initClassicParam(self):
        """
         В случае отсутствия данных из конфигурационного  файла используются стандарные значения
        """

        nanoTracking.STATIC_NONE_TRACK_ZONE_X = 500
        nanoTracking.STATIC_NONE_TRACK_ZONE_OTHER = 10
        nanoTracking.STATIC_FRAME_SKIP = 30
        nanoTracking.STATIC_FRAME_RESIZE_X = 3
        nanoTracking.STATIC_FRAME_RESIZE_Y = 4
        nanoTracking.STATIC_FRAME_RESIZE_BBOX = 4

        trackingAffine.STATIC_NONE_TRACK_ZONE_X = 500
        trackingAffine.STATIC_NONE_TRACK_ZONE_OTHER = 10
        trackingAffine.STATIC_FRAME_SKIP = 30
        trackingAffine.STATIC_FRAME_RESIZE_X = 3
        trackingAffine.STATIC_FRAME_RESIZE_Y = 4
        trackingAffine.STATIC_FRAME_RESIZE_BBOX = 4

        self.delay_frame = 0                    # ms
        self.fps = 80
        self.flag_with_delay = False
        self.deepSave = 1

    def setParams(self, param_dict):
        """
         При наличии конфигурационных данных
         :param param_dict:  dict - имеет в себе конфигурационные данные о зонах ограничения трекинга
        """
        nanoTracking.STATIC_NONE_TRACK_ZONE_X = param_dict['none_track_zone_x']
        nanoTracking.STATIC_NONE_TRACK_ZONE_OTHER = param_dict['none_track_zone_edge']
        nanoTracking.STATIC_FRAME_SKIP = param_dict['frame_skip']
        nanoTracking.STATIC_FRAME_RESIZE_X = param_dict['frame_resize_x']
        nanoTracking.STATIC_FRAME_RESIZE_Y = param_dict['frame_resize_y']
        nanoTracking.STATIC_FRAME_RESIZE_BBOX = param_dict['frame_resize_bbox']

        trackingAffine.STATIC_NONE_TRACK_ZONE_X = param_dict['none_track_zone_x']
        trackingAffine.STATIC_NONE_TRACK_ZONE_OTHER = param_dict['none_track_zone_edge']
        trackingAffine.STATIC_FRAME_SKIP = param_dict['frame_skip']
        trackingAffine.STATIC_FRAME_RESIZE_X = param_dict['frame_resize_x']
        trackingAffine.STATIC_FRAME_RESIZE_Y = param_dict['frame_resize_y']
        trackingAffine.STATIC_FRAME_RESIZE_BBOX = param_dict['frame_resize_bbox']

        self.delay_frame = param_dict['delay_frame']
        self.fps = param_dict['fps']
        if self.delay_frame >= 0:
            self.flag_with_delay = False
            self.deepSave = 1
        else:
            self.deepSave = math.ceil(self.delay_frame / (1000 / self.fps))
            self.flag_with_delay = True

    def start_tracking_surf(self, x, y, h, w, typeT = 0, d_frame=-1):
        """
         Функия настройки трекера на область с учетом задержки  кадра
         :param x: int  центр области по оси X
         :param y: int  центр области по оси Y
         :param h: int  высота области
         :param w: int  ширина области
         :param typeT: int  тип трекинга 0- афинные преобразования 1 - NN nano
         :param d_frame: int  задержка по количеству кадров
         :return: bool, При успешном инициализации области True
        """
        self.type = typeT
        if typeT == 0:
            self.track_affine.newPoint(x, y, h, w)
            try:
                self.track_affine.doit(self.get_frame_with_delay(d_frame))
                return True
            except cv.error as e:
                print('An error occurred:', e)
                self.__reinit__()
                return False
        else:
            self.track_nano.newPoint(x, y, h, w)
            try:
                self.track_nano.getTrack(self.get_frame_with_delay(d_frame))
                return True
            except cv.error as e:
                print('An error occurred:', e)
                self.__reinit__()
                return False


    def videoThreading(self, frame):
        """
         Функция обрабатывающая и распределяющая кадры между трекерами исходя из выбранного
         :param frame: np.array
         :return: bool (flag), np.array (frame), np.int32[4] (Rect)
        """
        if self.flag_with_delay:
            if len(self.frame_list) < self.deepSave:
                self.frame_list.append(frame)
            else:
                self.frame_list.remove(self.frame_list[0])
                self.frame_list.append(frame)
        elif len(self.frame_list) < 1:
            self.frame_list.append(frame)
        else:
            self.frame_list[0] = frame

        if self.type == 0:
            if self.track_affine.trackIsOn:
                return self.track_affine.doit(self.get_frame_with_delay())
            else:
                return False, frame, None
        elif self.type == 1:
            print('t', self.track_nano.trackIsOn)
            if self.track_nano.trackIsOn:
                return self.track_nano.getTrack(self.get_frame_with_delay())
            else:
                return False, frame, None
        else:
            return False, frame, None

    def get_frame_with_delay(self, d_frame=-1):
        """
        Функция обрабатывающая задержку по кадрам
        :param d_frame: количество кадров для пропуска
        :return: np.array кадр
        """
        return self.frame_list[0]

