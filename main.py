import os
import time

from config.config_manager import read_config
from feature_matcher import FeatureMatcher
import numpy as np
import cv2

SCOUT_IMG_PATH = 'test_imgs//300m//1 (15).jpg'
SEARCH_IMGS_FOLDER = 'test_imgs//aim1//search'
DIVE_IMGS_FOLDER = 'test_imgs//aim1//dive'
VIDEO_PATH = 'test_movies//search1.mp4'


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


def draw_matches(img1, kp1, img2, kp2):
    vis = combine_images_horizontally(img1, img2)

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

    def compare(self, img):
        s = time.time()
        img_kp, img_des = self.feature_matcher.detect(img)
        if img_des is None:
            return False, None, None
        kp1, kp2 = self.feature_matcher.match(
            self.scout_kp, self.scout_des, img_kp, img_des
        )
        print(time.time() - s)
        if kp1 is None or len(kp1) < 4:
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

    def start(self):
        cap = cv2.VideoCapture(VIDEO_PATH)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        idx = len([f for f in os.listdir('results') if f.endswith(".mp4")])
        out = cv2.VideoWriter(
            f'results/visualization_search_movie_{idx}.mp4',
            fourcc, 20.0,
            (self.scout_img.shape[1] * 2, self.scout_img.shape[0] * 2)
        )

        best_kp1 = []
        best_kp2 = []
        desired_view = None
        top_vis = np.zeros(self.scout_img.shape, np.uint8)
        bottom_vis = np.zeros(self.scout_img.shape, np.uint8)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:, 600:]
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            ret, kp1, kp2 = self.compare(frame)
            if not ret:
                continue

            if len(kp1) > len(best_kp1):
                best_kp1 = kp1
                best_kp2 = kp2
                desired_view = frame

                self.search_bbox = self.get_new_bbox(
                    self.feature_matcher.find_homography(best_kp2, best_kp1)
                )
                bottom_vis = combine_images_horizontally(
                    self.scout_img,
                    self.feature_matcher.transform_img(
                        cv2.rectangle(
                            desired_view.copy(),
                            self.search_bbox,
                            (0, 255, 255),
                            2
                        ),
                        self.feature_matcher.find_homography(kp1, kp2),
                        frame.shape[1],
                        frame.shape[0]
                    )
                )

            top_vis = draw_matches(
                cv2.rectangle(
                    self.scout_img,
                    self.object_bbox,
                    (0, 0, 255),
                    2
                ),
                kp1,
                frame,
                kp2
            )
            vis = combine_images_vertically(top_vis, bottom_vis)
            out.write(vis)
            # cv2.imshow('visualisation', resize_within_bounds(vis))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        out.release()
        # cv2.destroyWindow('visualisation')

        # H = self.feature_matcher.find_homography(best_kp2, best_kp1)
        # self.search_bbox = self.get_new_bbox(H)
        #
        # cv2.rectangle(desired_view, self.search_bbox, (0, 255, 255), 2)
        # cv2.imshow(
        #     'result',
        #     draw_matches(self.scout_img, best_kp1, desired_view, best_kp2)
        # )
        # cv2.waitKey(0)


def run():
    object_aim = ObjectAim()
    # object_aim.start_imgs()
    object_aim.start()


if __name__ == '__main__':
    run()
