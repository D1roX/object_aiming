# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import cv2, torch
import numpy as np
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import rect_2_cxy_wh, cxy_wh_2_rect


net = SiamRPNBIG()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNBIG.model')))
net.eval().cuda()
bbox_pts = None
state = None

video = cv2.VideoCapture('..//test_movies//popadanie.mpg')
if not video.isOpened():
    print("Не удалось открыть видеопоток")
    exit()

def new_point_wc(event, x, y, flags, param):
    global bbox_pts, state
    if event == cv2.EVENT_RBUTTONDOWN:
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = (np.array([x, y]), np.array([w, h]))
        target_pos, target_sz = bbox_pts
        state = SiamRPN_init(frame, target_pos, target_sz, net)

    if event == cv2.EVENT_LBUTTONDOWN:
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = (np.array([x, y]), np.array([w, h]))
        target_pos, target_sz = bbox_pts
        state = SiamRPN_init(frame, target_pos, target_sz, net)


cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Tracking", new_point_wc)

ret, prev_frame = video.read()

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = frame[:, :]

    if state is not None:
        state = SiamRPN_track(state, frame)
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]),
                      (0, 255, 255), 3)

    cv2.imshow("Tracking", frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()
