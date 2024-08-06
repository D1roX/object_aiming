import math
import time

import torch.cuda

from config.config_manager import read_config
from feature_matcher import FeatureMatcher

from tracker import Tracker
from utils import *

SCOUT_IMG_PATH = 'test_imgs//merge.jpg'
SEARCH_IMGS_FOLDER = 'test_imgs//aim1//search'
DIVE_IMGS_FOLDER = 'test_imgs//aim1//dive'
VIDEO_PATH = 'test_movies//FL+5.mp4'
SKIP_FRAME = 250


class ObjectAim:
    def __init__(self):
        cfg = read_config()
        self.feature_matcher = FeatureMatcher(cfg['feature_matcher'])
        self.scout_img = cv2.imread(SCOUT_IMG_PATH)
        self.object_img, self.object_bbox = select_object(self.scout_img)
        self.scout_kp, self.scout_des = self.feature_matcher.detect(
            self.scout_img
        )
        self.prev_kp = None
        self.prev_des = None
        self.search_bbox = None
        self.bbox_confidence = 0
        self.tracker = Tracker('nano')
        self.track_ready = False
        self.bbox = None
        self.cap = cv2.VideoCapture(VIDEO_PATH)

    def calc_scale(self, cur_kp, cur_des):
        if self.prev_kp is None:
            self.prev_kp = cur_kp
            self.prev_des = cur_des
            return
        kp1, kp2 = self.feature_matcher.match(
            self.prev_kp, self.prev_des, cur_kp, cur_des
        )
        angle_rotated_searcher(kp1, kp2)

    def compare(self, img):
        img_kp, img_des = self.feature_matcher.detect(img)
        if img_des is None:
            return False, None, None
        self.calc_scale(img_kp, img_des)
        kp1, kp2 = self.feature_matcher.match(
            self.scout_kp, self.scout_des, img_kp, img_des
        )
        if kp1 is None or len(kp1) < 8:
            return False, None, None
        return True, kp1, kp2

    def get_bbox(self, frame, kp1, kp2):
        if 0 <= self.bbox_confidence < 5:
            self.bbox_confidence += 1
            return None
        return check_bbox_bounds(
            transform_bbox(
                self.object_bbox,
                self.feature_matcher.find_homography(kp1, kp2)
            ),
            frame.shape[1],
            frame.shape[0],
        )

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def start(self):
        total_frame = 0
        while True:
            frame = self.get_frame()
            if frame is None:
                break

            total_frame += 1
            if total_frame < SKIP_FRAME:
                continue

            ok, kp1, kp2 = self.compare(frame)

            vis = resize_within_bounds(
                draw_matches(
                    cv2.rectangle(self.scout_img, self.object_bbox,
                                  (255, 0, 0), 2),
                    kp1,
                    frame,
                    kp2
                )
            )
            cv2.imshow('vis', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not ok:
                self.bbox_confidence = 0
                continue
            else:
                self.bbox = self.get_bbox(frame, kp1, kp2)
                if self.bbox is not None:
                    self.track_ready = True
                    self.tracker.start_tracking(frame, self.bbox)
                    break

        while True:
            frame = self.get_frame()
            if frame is None:
                break
            ok, self.bbox = self.tracker.update(frame)
            vis = resize_within_bounds(
                combine_images_horizontally(
                    cv2.rectangle(self.scout_img, self.object_bbox,
                                  (255, 0, 0), 2),
                    cv2.rectangle(frame, self.bbox, (0, 255, 255), 2)
                )
            )
            cv2.imshow('vis', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()


def run():
    object_aim = ObjectAim()
    object_aim.start()


if __name__ == '__main__':
    run()
    # print(torch.cuda.is_available())