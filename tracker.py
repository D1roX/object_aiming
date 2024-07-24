import cv2
import numpy as np

from config.config_manager import read_config
from feature_matcher import FeatureMatcher


class Tracker:
    def __init__(self, tracker_type='nano'):
        self.tracker_type = tracker_type
        if tracker_type == 'nano':
            self.tracker = NanoTracker()
        elif tracker_type == 'kp':
            self.tracker = KPTracker()
        elif tracker_type == 'of':
            pass

        self.bbox = None
        self.state = False

    def start_tracking(self, frame, bbox):
        self.bbox = bbox
        self.state = True
        return self.tracker.start_tracking(frame, self.bbox)

    def update(self, frame):
        ok, self.bbox = self.tracker.update(frame)
        return ok, self.bbox

    def get_bbox(self):
        return self.bbox

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


class KPTracker:
    def __init__(self):
        cfg = read_config()
        self.feature_matcher = FeatureMatcher(cfg['feature_matcher'])
        self.cur_kp = None
        self.cur_des = None
        self.bbox = None

    def _calc_new_bbox(self, bbox, H):
        x, y, w, h = bbox
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        pts = self.feature_matcher.transform_points(pts, H)
        pts = pts.reshape(-1, 2).astype(np.int32)

        min_x = np.min(pts[:, 0])
        # min_x1 = np.min(pts[:2, 0])
        # min_x2 = np.min(pts[2:, 0])
        # min_x = max(min_x1, min_x2)

        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        return min_x, min_y, max_x - min_x, max_y - min_y

    def start_tracking(self, frame, bbox):
        self.bbox = bbox
        self.cur_kp, self.cur_des = self.feature_matcher.detect(frame)
        self.feature_matcher._cfg['match_conf'] = 0.4

    def update(self, frame):
        new_kp, new_des = self.feature_matcher.detect(frame)
        kp1, kp2 = self.feature_matcher.match(
            self.cur_kp, self.cur_des, new_kp, new_des
        )
        if kp1 is None or len(kp1) < 8:
            print('Недостаточно совпадений')
            return False, self.bbox
        # img = self.feature_matcher.transform_img(frame, self.feature_matcher.find_homography(kp1, kp2), frame.shape[1], frame.shape[0])
        # cv2.imshow('1', frame)
        # cv2.imshow('2', img)
        # cv2.waitKey(0)
        self.cur_kp = new_kp
        self.cur_des = new_des
        H = self.feature_matcher.find_homography(kp2, kp1)
        self.bbox = self._calc_new_bbox(self.bbox, H)
        return True, self.bbox
