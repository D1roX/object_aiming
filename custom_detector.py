import cv2
import numpy as np
import torch


class CustomDetector:
    def __init__(self, **conf):
        self.detector = cv2.ORB_create()
    def run_opencv_detector(self, image):
        detections, descriptors = self.detector.detectAndCompute(image, None)
        points = np.array([k.pt for k in detections], dtype=np.float32)
        scores = np.array([k.response for k in detections], dtype=np.float32)
        scales = np.array([k.size for k in detections], dtype=np.float32)
        angles = np.deg2rad(
            np.array([k.angle for k in detections], dtype=np.float32))
        return points, scores, scales, angles, descriptors

    def extract(self, image):
        keypoints, scores, scales, angles, descriptors = self.run_opencv_detector(
            (image * 255.0).astype(np.uint8)
        )
        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": angles,
            "descriptors": descriptors,
        }
        if scores is not None:
            pred["keypoint_scores"] = scores

        pred = {k: torch.from_numpy(v) for k, v in pred.items()}
        if scores is not None:
            num_points = 1024
            if num_points is not None and len(pred["keypoints"]) > num_points:
                indices = torch.topk(pred["keypoint_scores"],
                                     num_points).indices
                pred = {k: v[indices] for k, v in pred.items()}

        return pred
