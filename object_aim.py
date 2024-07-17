import os
import time

from config.config_manager import read_config
from feature_matcher import FeatureMatcher
import numpy as np
import cv2

from tracker.main_track import mainTrack

SCOUT_IMG_PATH = 'test_imgs//300m//1 (12).jpg'
SEARCH_IMGS_FOLDER = 'test_imgs//aim1//search'
DIVE_IMGS_FOLDER = 'test_imgs//aim1//dive'
VIDEO_PATH = 'test_movies//search2.mp4'


def select_object(img):
    x, y, w, h = cv2.selectROI('select object', img)
    cv2.destroyWindow('select object')
    return img[y: y + h, x: x + w], (x, y, w, h)


def resize_within_bounds(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio)
    if scale_factor >= 1:
        return image
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def check_bbox_bounds(bbox, max_w, max_h):
    x, y, w, h = bbox
    return (0 <= x <= max_w and 0 <= y <= max_h
            and x < x + w <= max_w and y < y + h <= max_h)


def draw_matches(img1, kp1, img2, kp2):
    vis = combine_images_horizontally(img1, img2)
    if kp1 is None or kp2 is None:
        return vis
    for kp1, kp2 in zip(kp1, kp2):
        pt1 = (int(kp1[0][0] * 3), int(kp1[0][1] * 3))
        pt2 = (int(kp2[0][0] * 3) + img1.shape[1], int(kp2[0][1] * 3))

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)

    return vis


def combine_images_horizontally(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1, :3] = img1
    vis[:h2, w1:w1 + w2, :3] = img2

    return vis


def combine_images_vertically(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
    vis[:h1, :w1, :3] = img1
    vis[h1:h1 + h2, :w2, :3] = img2

    return vis


class ObjectAim:
    def __init__(self):
        cfg = read_config()
        self.feature_matcher = FeatureMatcher(cfg['feature_matcher'])
        self.scout_img = cv2.imread(SCOUT_IMG_PATH)[:, 600:]
        self.object_img, self.object_bbox = select_object(self.scout_img)

        self.scout_kp, self.scout_des = self.feature_matcher.detect(
            self.scout_img
        )
        self.search_bbox = None

        self.bbox_confidence = 0

        self.tracker = mainTrack()

    def compare(self, img):
        img_kp, img_des = self.feature_matcher.detect(img)
        if img_des is None:
            return False, None, None
        kp1, kp2 = self.feature_matcher.match(
            self.scout_kp, self.scout_des, img_kp, img_des
        )
        if kp1 is None or len(kp1) < 8:
            return False, None, None
        return True, kp1, kp2

    def get_new_bbox(self, H):
        x, y, w, h = self.object_bbox
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        pts = self.feature_matcher.transform_points(pts, H)
        pts = pts.reshape(-1, 2).astype(np.int32)
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        return min_x, min_y, max_x - min_x, max_y - min_y

    def start_imgs(self):
        best_kp1 = []
        best_kp2 = []
        desired_view = None
        for filename in os.listdir(SEARCH_IMGS_FOLDER):
            img = cv2.imread(os.path.join(SEARCH_IMGS_FOLDER, filename))[:,
                  600:]
            ret, kp1, kp2 = self.compare(img)
            if not ret:
                continue
            if len(kp1) > len(best_kp1):
                best_kp1 = kp1
                best_kp2 = kp2
                desired_view = img
        if desired_view is None:
            raise Exception('No object detected')

        H = self.feature_matcher.find_homography(best_kp2, best_kp1)
        H2 = self.feature_matcher.find_homography(best_kp1, best_kp2)

        ####
        aligned_img = self.feature_matcher.transform_img(
            desired_view, H2, desired_view.shape[1], desired_view.shape[0]
        )
        align_vis = combine_images_horizontally(self.scout_img, aligned_img)
        ####

        self.search_bbox = self.get_new_bbox(H)
        cv2.rectangle(desired_view, self.search_bbox, (0, 255, 255), 2)
        matches_vis = draw_matches(
            cv2.rectangle(self.scout_img, self.object_bbox, (0, 0, 255), 2),
            best_kp1,
            desired_view,
            best_kp2
        )

        vis = combine_images_vertically(matches_vis, align_vis)
        idx = len([f for f in os.listdir('results') if f.endswith(".jpg")])
        cv2.imwrite(f'results/visualization_search_img_{idx}.jpg', vis)
        return desired_view, best_kp1, best_kp2

    def get_bbox(self, frame, kp1, kp2):
        if 0 <= self.bbox_confidence < 5:
            self.bbox_confidence += 1
            return None

        return self.get_new_bbox(
            self.feature_matcher.find_homography(kp2, kp1)
        )

    def start(self):
        cap = cv2.VideoCapture(VIDEO_PATH)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        idx = len([
            f for f in os.listdir('results') if (
                f.endswith(".mp4") and f.startswith("tracking")
            )
        ])
        out = cv2.VideoWriter(
            f'results/tracking_{idx}.mp4',
            fourcc, 20.0,
            (self.scout_img.shape[1] * 2, self.scout_img.shape[0])
        )

        total_frame = 0

        is_object_detected = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frame += 1
            if total_frame < 150:
                continue

            frame = frame[:, 600:]

            bbox_ok, _, self.search_bbox = self.tracker.videoThreading(
                frame
            )

            ret, kp1, kp2 = self.compare(frame)
            vis = draw_matches(
                cv2.rectangle(
                    self.scout_img,
                    self.object_bbox,
                    (0, 0, 255),
                    2
                ),
                kp1,
                cv2.rectangle(
                    frame,
                    self.search_bbox,
                    (0, 255, 255),
                    2
                ),
                kp2,
            )
            cv2.imshow('visualisation', resize_within_bounds(vis))
            out.write(vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if not ret:
                self.bbox_confidence = 0
                continue

            if not is_object_detected:
                init_bbox = self.get_bbox(frame, kp1, kp2)
                print(init_bbox)
                if not init_bbox or not check_bbox_bounds(
                        init_bbox, frame.shape[1], frame.shape[0]
                ):
                    continue
                # cv2.rectangle(frame, init_bbox, (0, 255, 255), 2)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                self.tracker.start_tracking_surf(
                    init_bbox[0] + init_bbox[3] / 2,
                    init_bbox[1] + init_bbox[2] / 2,
                    init_bbox[3],
                    init_bbox[2],
                    1)
                is_object_detected = True
                print('FOUND')
            else:
                if not bbox_ok:
                    print('OBJECT LOST')
                    self.bbox_confidence = 0
                    is_object_detected = False
                    continue
        cap.release()
        out.release()


def run():
    object_aim = ObjectAim()
    # object_aim.start_imgs()
    object_aim.start()


if __name__ == '__main__':
    run()
