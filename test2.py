import time

import torch

from aliked import ALIKED
from disk import DISK
from lightglue import LightGlue
from sift import SIFT
from superpoint import SuperPoint
from dog_hardnet import DoGHardNet
from uutils import load_image, rbd
import cv2
import numpy as np


extractor = SuperPoint(max_num_keypoints=128).eval()
matcher = LightGlue(features='superpoint').eval()
bbox_pts = None

video = cv2.VideoCapture('test_movies//popadanie.mpg')
if not video.isOpened():
    print("Не удалось открыть видеопоток")
    exit()


def new_point_wc(event, x, y, flags, param):
    global bbox_pts
    if event == cv2.EVENT_RBUTTONDOWN:
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32).reshape(-1, 1, 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32).reshape(-1, 1, 2)


cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Tracking", new_point_wc)

ret, prev_frame = video.read()

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = frame[:, :]

    feats = extractor.extract(t_frame)
    t = time.time()
    matches01 = matcher({"image0": prev_feats, "image1": feats})
    print('matcher time: ', time.time() - t)
    feats0, feats1, matches01 = [
        rbd(x) for x in [prev_feats, feats, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], \
    matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    m_kpts0 = m_kpts0.cpu().numpy().astype(np.float32)
    m_kpts1 = m_kpts1.cpu().numpy().astype(np.float32)
    M, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
    if bbox_pts is not None:
        transformed_bbox = cv2.perspectiveTransform(bbox_pts, M)

        transformed_bbox_int = np.int32(transformed_bbox)
        cv2.rectangle(frame, tuple(transformed_bbox_int[0, 0]),
                      tuple(transformed_bbox_int[2, 0]), (0, 255, 0), 2)

        # Обновление точек рамки для следующей итерации
        bbox_pts = transformed_bbox

    # Отображение результата
    cv2.imshow("Tracking", frame)

    # Обновление предыдущего кадра
    prev_frame = t_frame
    prev_frame = feats1

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()
