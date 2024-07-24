import cv2
import numpy as np

import video
import main_track
import trackingAffine
import nanoTracking
import vitTracking

mt = main_track.mainTrack()

def newPointWC(event, x, y, flags, param):
    """
        Обработка события OpenCV
        :param event: Событие
        :param x: координаты OX
        :param y: координаты OY
        :param flags: не используется
        :param param: не используется
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        mt.start_tracking_surf(x, y, 60, 60, 0) # AF

    if event == cv2.EVENT_RBUTTONDOWN:
        mt.start_tracking_surf(x, y, 60, 60, 1) # NN

win = cv2.namedWindow("mainWin")
cv2.setMouseCallback("mainWin", newPointWC)

def drawTrackPoint(frame, point2T, wRect, hRect, th,color = (255,0,0)):
    """
        Отрисовка точки трекинга
        :param frame: кадр на котором изображается маркер
       """
    cv2.rectangle(frame, (int(point2T[0] - wRect / 2), int(point2T[1] - hRect / 2)),
                  (int(point2T[0] + wRect / 2), int(point2T[1] + hRect / 2)), color, th)

    cv2.line(frame, (int(point2T[0] - wRect / 4), int(point2T[1])),
             (int(point2T[0] + wRect / 4), int(point2T[1])), color, th)

    cv2.line(frame, (int(point2T[0]), int(point2T[1] - hRect / 4)),
             (int(point2T[0]), int(point2T[1] + hRect / 4)), color,  th)

def drawTrackPoint2(frame, rect, th,color = (255,0,0)):
    """
        Отрисовка точки трекинга v 2
        :param frame: кадр на котором изображается маркер
       """
    cv2.rectangle(frame, rect, color, th)



def testTracking():
    ######################### DEBUG #############################
    vid = video.video("videos/popadanie.mpg", stab=False)
    while True:
        frame = vid.grab_frame()        # получение кадра, заменить на кадр из потока FincoPlayer
        if frame is not None:
            res, frame_res, rect = mt.videoThreading(frame)
            if res == True:
                #drawTrackPoint2(frame, (rect[0]-np.int32(rect[2]/2), rect[1]-np.int32(rect[3]/2)), rect[2], rect[3], 4, color=(255, 0, 0))
                #drawTrackPoint(frame, point2, track.wRect*4, track.hRect*4, track.th, color=(0, 255, 0))                                 # закрасить участок трекинга
                #drawTrackPoint2(frame2, tr_nano.point2T, int(tr_nano.hw[0]), int(tr_nano.hw[1]), 4, color=(255, 0, 0))
                drawTrackPoint2(frame, rect, 2, color=(0, 255, 0))
                #cv2.imshow("mainWin2", frame_res)
            cv2.imshow("mainWin", frame)
            print('Result'+str(res))
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        else:
            return




if __name__ == '__main__':
    testTracking()

