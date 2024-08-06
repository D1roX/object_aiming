import math

import cv2
import numpy as np

from config.config_manager import read_config
from feature_matcher import FeatureMatcher
from utils import angle_rotated_searcher, transform_bbox


class Tracker:
    def __init__(self, tracker_type='nano'):
        self.tracker_type = tracker_type
        if tracker_type == 'nano':
            self.tracker = NanoTracker()
        elif tracker_type == 'superpoint':
            self.tracker = AffineTracker()
        elif tracker_type == 'orb':
            self.tracker = ORBTracker()

        self.bbox = None
        self.state = False

    def switch_tracker(self, frame):
        square = (self.bbox[2] * self.bbox[3] * 100) / (frame.shape[0] * frame.shape[1])
        print(f'Object square: {square:.2f}')
        if self.tracker_type != 'nano':
            return
        if square >= 0.7:
            self.tracker_type = 'orb'
            print('\n\nSwitched to ORBTracker\n')
            self.tracker = ORBTracker()
            self.tracker.start_tracking(frame, list(self.bbox))

    def start_tracking(self, frame, bbox):
        self.bbox = bbox
        self.state = True
        return self.tracker.start_tracking(frame, self.bbox)

    def update(self, frame):
        ok, self.bbox = self.tracker.update(frame)
        self.switch_tracker(frame)
        return ok, self.bbox

    def get_state(self):
        return self.state


class NanoTracker:
    def __init__(self):
        params = cv2.TrackerNano.Params()
        params.backbone = 'models/nanotrack_backbone_sim.onnx'
        params.neckhead = 'models/nanotrack_head_sim.onnx'
        self.tracker = cv2.TrackerNano.create(params)

    def start_tracking(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def update(self, frame):
        return self.tracker.update(frame)


class AffineTracker:
    def __init__(self):
        cfg = read_config()
        self.feature_matcher = FeatureMatcher(cfg['feature_matcher'])
        self.cur_kp = None
        self.cur_des = None
        self.bbox = None

        self.prev_angle = 0
        self.scale = None

    def start_tracking(self, frame, bbox):
        self.bbox = bbox
        self.cur_kp, self.cur_des = self.feature_matcher.detect(frame)
        # self.feature_matcher._cfg['match_conf'] = 0.4

    def delta_check(self, ret):
        x = (self.bbox[0] * ret[0][0] + self.bbox[1] * ret[0][1] +
             self.scale * ret[0][2])
        y = (self.bbox[0] * ret[1][0] + self.bbox[1] * ret[1][1] +
             self.scale * ret[1][2])
        self.bbox[0] = x
        self.bbox[1] = y
        self.bbox[2] = self.bbox[2] * self.scale
        self.bbox[3] = self.bbox[3] * self.scale

        # print(self.bbox)
        # self.bbox = transform_bbox(self.bbox, ret)
        # print(self.bbox)

    def update(self, frame):
        new_kp, new_des = self.feature_matcher.detect(frame)
        kp1, kp2 = self.feature_matcher.match(
            self.cur_kp, self.cur_des, new_kp, new_des, max_matches=90
        )
        if kp1 is None or len(kp1) < 8:
            print('Недостаточно совпадений')
            return False, self.bbox

        ret, inliners, self.scale, self.prev_angle = angle_rotated_searcher(
            kp1, kp2
        )
        # self.delta_check(ret)

        self.cur_kp = new_kp
        self.cur_des = new_des
        # print(self.bbox)
        self.bbox = transform_bbox(
            self.bbox, self.feature_matcher.find_homography(kp1, kp2)
        )
        # print(self.bbox)
        return True, (
            int(self.bbox[0]),
            int(self.bbox[1]),
            int(self.bbox[2]),
            int(self.bbox[3])
        )


class ORBTracker:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.detector = cv2.ORB_create()
        self.cur_kp = None
        self.cur_des = None
        self.bbox = None

        self.prev_angle = 0
        self.scale = None

    def match(self, new_kp, new_des):
        matches = self.matcher.knnMatch(self.cur_des, new_des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)
        src_pts = np.float32(
            [self.cur_kp[m.queryIdx].pt for m in good]).reshape(-1, 1,
                                                                2)
        dst_pts = np.float32([new_kp[m.trainIdx].pt for m in good]).reshape(-1,
                                                                            1,
                                                                            2)
        return src_pts, dst_pts

    def start_tracking(self, frame, bbox):
        self.bbox = bbox
        self.cur_kp, self.cur_des = self.detector.detectAndCompute(frame, None)

    def update(self, frame):
        new_kp, new_des = self.detector.detectAndCompute(frame, None)
        kp1, kp2 = self.match(
            new_kp, new_des
        )
        if kp1 is None or len(kp1) < 8:
            print('Недостаточно совпадений')
            return False, self.bbox

        ret, inliners, self.scale, self.prev_angle = angle_rotated_searcher(
            kp1, kp2
        )

        self.cur_kp = new_kp
        self.cur_des = new_des
        H, _ = cv2.findHomography(
            kp1, kp2, cv2.RANSAC, 10.0, confidence=0.99, maxIters=2000
        )
        self.bbox = transform_bbox(
            self.bbox, H
        )
        return True, (
            int(self.bbox[0]),
            int(self.bbox[1]),
            int(self.bbox[2]),
            int(self.bbox[3])
        )
