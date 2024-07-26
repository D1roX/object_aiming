import traceback

import cv2
import numpy as np

import tools
from exceptions import FeatureMatcherException
from logger import Logger
from models.super_point import SuperPoint

logger = Logger(__name__)


class FeatureMatcher:
    """
    Класс для сопоставления признаков на двух изображениях.
    """

    def __init__(self, cfg: dict):
        """
        Инициализация объекта FeatureMatcher.

        :param cfg: Словарь с параметрами алгоритма.
        """
        self._cfg = cfg
        if self._cfg['detector'] == 'superpoint':
            self._detector = SuperPoint(**self._cfg['super_point'])
        else:
            self._detector = cv2.AKAZE_create()
        self._matcher = cv2.FlannBasedMatcher(
            self._cfg['flann_index_params'], self._cfg['flann_search_params']
        )

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения перед поиском признаков.

        :param img: Изображение для предобработки.

        :return: Предобработанное изображение.
        """
        img = tools.resize(
            img, fx=self._cfg['img_size_scale'], fy=self._cfg['img_size_scale']
        )
        img = tools.convert_color(img)
        try:
            if not isinstance(img, np.float32):
                return img.astype(np.float32) / 255.0
            return img / 255.0
        except np.linalg.LinAlgError:
            logger.error(
                'Ошибка при подготовке изображения.\n' + traceback.format_exc()
            )
            raise FeatureMatcherException()

    def detect(self, img):
        img = self.preprocess_image(img)
        kp1, des1, _ = self._detector.detect_and_compute(img)
        return kp1, des1

    def match(self, kp1, des1, kp2, des2, max_matches=40):
        try:
            matches = self._matcher.knnMatch(des1.T, des2.T, k=2)
            # matches = sorted(
            #     matches,
            #     key=lambda x: x[0].distance
            # )[:max_matches]
            good_matches = []
            for m, n in matches:
                if m.distance < n.distance * self._cfg['match_conf']:
                    good_matches.append((m.trainIdx, m.queryIdx))
            if not good_matches:
                return None, None
            kp1 = kp1[:2, np.array(good_matches)[:, 1]].T.reshape(-1, 1, 2)
            kp2 = kp2[:2, np.array(good_matches)[:, 0]].T.reshape(-1, 1, 2)
            return (
                kp1[: self._cfg['max_features']],
                kp2[: self._cfg['max_features']],
            )
        except (IndexError, cv2.error) as _:
            logger.error(
                'Ошибка при вычислении особых точек изображения.\n'
                + traceback.format_exc()
            )
            raise FeatureMatcherException()

    def resize_homography(self, H: np.ndarray) -> np.ndarray:
        """
        Масштабирование матрицы гомографии обратно к исходному размеру
        изображения.

        :param H: Матрица гомографии.

        :return: Масштабированная матрица гомографии.
        """
        scaling_matrix = np.array(
            [
                [self._cfg['img_size_scale'], 0, 0],
                [0, self._cfg['img_size_scale'], 0],
                [0, 0, 1],
            ]
        )
        inverse_scaling_matrix = np.array(
            [
                [1 / self._cfg['img_size_scale'], 0, 0],
                [0, 1 / self._cfg['img_size_scale'], 0],
                [0, 0, 1],
            ]
        )
        return inverse_scaling_matrix @ H @ scaling_matrix

    def find_homography(self, kp1: np.ndarray, kp2: np.ndarray) -> np.ndarray:
        """
        Нахождение матрицы гомографии, которая преобразует особые точки второго
        изображения в особые точки первого изображения.

        :param kp1: Ключевые точки на первом изображении.
        :param kp2: Ключевые точки на втором изображении.

        :return: Матрица гомографии.
        """
        try:
            return self.resize_homography(
                cv2.findHomography(
                    kp1, kp2, cv2.RANSAC,
                    self._cfg['ransac_reproj_threshold'],
                    maxIters=self._cfg['ransac_maxIters'],
                    confidence=self._cfg['ransac_confidence']
                )[0]
            )
        except cv2.error as _:
            logger.error(
                'Ошибка при поиске гомографии.\n' + traceback.format_exc()
            )
            raise FeatureMatcherException()

    def transform_img(
        self, img: np.ndarray, H: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """
        Трансформация изображения с помощью матрицы гомографии.

        :param img: Изображение для трансформации.
        :param H: Матрица гомографии.
        :param w: Ширина выходного изображения.
        :param h: Высота выходного изображения.

        :return: Трансформированное изображение.
        """
        try:
            if H.shape == (3, 3):
                return cv2.warpPerspective(
                    img, H, (w, h), flags=cv2.INTER_NEAREST
                )
            return cv2.warpAffine(img, H, (w, h))
        except cv2.error as _:
            logger.error(
                'Ошибка при трансформации изображения.\n'
                + traceback.format_exc()
            )
            raise FeatureMatcherException()

    def transform_points(self, pts: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Трансформация точек с помощью матрицы гомографии.

        :param pts: Точки для трансформации.
        :param H: Матрица гомографии.

        :return: Трансформированные точки.
        """
        try:
            if H.shape == (3, 3):
                return cv2.perspectiveTransform(pts, H)
            return cv2.transform(pts, H)
        except cv2.error as _:
            logger.error(
                'Ошибка при трансформации точек.\n' + traceback.format_exc()
            )
            raise FeatureMatcherException()

    def set_config(self, config: dict) -> None:
        """
        Замена словаря с параметрами алгоритма.

        :param config: Новый словарь с параметрами алгоритма.
        """
        self._cfg = config
