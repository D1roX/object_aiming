import math

import numpy as np
import cv2


def select_object(img):
    x, y, w, h = cv2.selectROI('select object', img)
    cv2.destroyWindow('select object')
    return img[y: y + h, x: x + w], (x, y, w, h)


def resize_within_bounds(image, max_width=2560, max_height=1440):
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
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if (0 <= x <= max_w and 0 <= y <= max_h
            and x < x + w <= max_w and y < y + h <= max_h):
        return [x, y, w, h]
    return None


def draw_matches(img1, kp1, img2, kp2):
    vis = combine_images_horizontally(img1, img2)
    if kp1 is None or kp2 is None:
        return vis
    for kp1, kp2 in zip(kp1, kp2):
        pt1 = (int(kp1[0][0] * 2), int(kp1[0][1] * 2))
        pt2 = (int(kp2[0][0] * 2) + img1.shape[1], int(kp2[0][1] * 2))

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


def transform_bbox(bbox, H):
    x, y, w, h = bbox
    pts = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, H) if H.shape == (3, 3) else (
        cv2.transform(pts, H))
    pts = pts.reshape(-1, 2)
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def angle_rotated_searcher(prev_pts, curr_pts):
    ret, inliners = cv2.estimateAffinePartial2D(
        prev_pts, curr_pts, method=cv2.RANSAC, confidence=0.99
    )
    if ret is None:
        return ret, inliners
    angle = math.atan((-ret[0][1]) / ret[0][0])
    scale = ret[0][0] / math.cos(angle)
    print('scale: ', scale)
    return ret, inliners, scale, angle
