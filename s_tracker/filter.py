import numpy as np

class stabilizer:
    """
        Класс для стабилизации шумов, метод усреднения
    """
    def  __init__(self, h=0, w=0):
        """
         Инициализация класса
         :param h:
         :param w:
         :param stab_k:
         :param oldPoint:
         :param point:
         :param predictPoint:
         :param hw:
        """
        self.stab_k = 0.1
        self.oldPoint = self.point = self.predictPoint = np.zeros(2, np.int32)
        self.hw = [h, w]


    def clr(self):
        """
         Очистка значений при инициализации
        :return :None
        """
        self.oldPoint = np.zeros(2, np.int32)
        self.point = np.zeros(2, np.int32)
        self.predictPoint = np.zeros(2, np.int32)

    def sethw(self, h, w):
        """
         Сеттер допустимых отклонений
        :param h: int
        :param w: int
        :return :None
        """
        self.hw[0] = h
        self.hw[1] = w

    def sethw(self, hw):
        """
         Сеттер допустимых отклонений
        :param hw: int[2]
        :return :None
        """
        self.hw = hw

    def predict(self, point):
        """
         Рассчет усреднененной позиции
        :param point: int[2]
        :return : bool, int[2]
        """
        self.point = point
        if self.oldPoint[0] == 0:
            self.oldPoint = point
            return False, point
        else:
            if self.oldPoint[0]+self.hw[0] > self.point[0] and self.oldPoint[0]-self.hw[0] < self.point[0] and self.oldPoint[1]+self.hw[1] > self.point[1] and self.oldPoint[1] - self.hw[1] < self.point[1]:
                self.predictPoint = (self.point + self.oldPoint)/2
                self.oldPoint = self.point
                return True, self.predictPoint
            else:
                return False, point