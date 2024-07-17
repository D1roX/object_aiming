import os
import sys
import traceback
from functools import partial
from multiprocessing import Event, Manager, Pool, Semaphore, cpu_count
from threading import Thread

import cv2
import numpy as np

from exceptions import ImageDifferenceSearcherException, CBaseException
from logger import Logger

logger = Logger(__name__)


def abs_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Вычисление абсолютного различия между двумя изображениями.

    :param img1: Первое изображение.
    :param img2: Второе изображение.

    :return: Изображение абсолютного различия.
    """
    try:
        return cv2.absdiff(img1, img2)
    except cv2.error as _:
        logger.error(
            'Ошибка при вычислении абсолютной разницы.\n'
            + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def binary_difference(img: np.ndarray, thresh, max_val):
    """
    Бинаризация изображения.

    :param img: Изображение.
    :param thresh: Порог бинаризации.
    :param max_val: Максимальное значение пикселя.

    :return: Бинаризированное изображение.
    """
    try:
        return cv2.threshold(img, thresh, max_val, cv2.THRESH_BINARY)[1]
    except cv2.error as _:
        logger.error(
            'Ошибка при бинаризации изображений.\n' + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def merge_rectangles(rect1: tuple, rect2: tuple) -> tuple:
    """
    Объединяет два прямоугольника в один.

    :param rect1: Первый прямоугольник (x1, y1, w1, h1).
    :param rect2: Второй прямоугольник (x2, y2, w2, h2).

    :return: Объединенный прямоугольник.
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - min_x
    h = max(y1 + h1, y2 + h2) - min_y
    return min_x, min_y, w, h


def check_overlap(
    x1: int, y1: int, w1: int, h1: int, x2: int, y2: int, w2: int, h2: int
) -> bool:
    """
    Проверка на пересечение двух прямоугольников.

    :param x1: Координата x верхнего левого угла первого прямоугольника.
    :param y1: Координата y верхнего левого угла первого прямоугольника.
    :param w1: Ширина первого прямоугольника.
    :param h1: Высота первого прямоугольника.
    :param x2: Координата x верхнего левого угла второго прямоугольника.
    :param y2: Координата y верхнего левого угла второго прямоугольника.
    :param w2: Ширина второго прямоугольника.
    :param h2: Высота второго прямоугольника.

    :return: True, если прямоугольники пересекаются, иначе - False.
    """
    return not (x2 > x1 + w1 or x1 > x2 + w2 or y2 > y1 + h1 or y1 > y2 + h2)


def merge_overlapping_contours(contours: list) -> list:
    """
    Объединяет пересекающиеся контуры в один.

    :param contours: Список прямоугольников.

    :return: Список объединенных прямоугольников.
    """
    try:
        merged_contours = []
        while contours:
            contour = contours.pop(0)
            x, y, w, h = contour
            has_overlap = False
            for i, other_contour in enumerate(contours):
                ox, oy, ow, oh = other_contour
                if check_overlap(x, y, w, h, ox, oy, ow, oh):
                    contours[i] = merge_rectangles(contour, other_contour)
                    has_overlap = True
                    break
            if not has_overlap:
                merged_contours.append(contour)
        return merged_contours
    except Exception as _:
        logger.error(
            'Ошибка при объединении контуров.\n' + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def find_rect_corners(
    w: int, h: int, H: np.ndarray, shift_x: int = 0, shift_y: int = 0
) -> tuple[int, int, int, int]:
    """
    Находит координаты углов максимального прямоугольника, вмещающего
    преобразованное с помощью матрицы гомографии изображение без пустых
    пикселей (появляются в результате трансформации).

    :param w: Ширина прямоугольника.
    :param h: Высота прямоугольника.
    :param H: Матрица гомографии.
    :param shift_x: Смещение по оси x.
    :param shift_y: Смещение по оси y.

    :return: Координаты углов прямоугольника.
    """
    try:
        if H.shape == (3, 3):
            top_left = H.dot(np.array([shift_x, shift_y, 1]))
            top_right = H.dot(np.array([w + shift_x, shift_y, 1]))
            bottom_right = H.dot(np.array([w + shift_x, h + shift_y, 1]))
            bottom_left = H.dot(np.array([shift_x, h + shift_y, 1]))

            top_left /= top_left[2]
            top_right /= top_right[2]
            bottom_right /= bottom_right[2]
            bottom_left /= bottom_left[2]
        else:
            pts = np.float32(
                [
                    [shift_x, shift_y],
                    [w + shift_x, shift_y],
                    [w + shift_x, h + shift_y],
                    [shift_x, h + shift_y],
                ]
            ).reshape(-1, 1, 2)
            transformed_pts = cv2.transform(pts, H)
            top_left = transformed_pts[0, 0]
            top_right = transformed_pts[1, 0]
            bottom_right = transformed_pts[2, 0]
            bottom_left = transformed_pts[3, 0]
        top_left = top_left.astype(int)
        top_right = top_right.astype(int)
        bottom_right = bottom_right.astype(int)
        bottom_left = bottom_left.astype(int)

        min_x = max(top_left[0], bottom_left[0], shift_x) - shift_x
        max_x = min(top_right[0], bottom_right[0], w + shift_x) - shift_x
        min_y = max(top_left[1], top_right[1], shift_y) - shift_y
        max_y = min(bottom_left[1], bottom_right[1], h + shift_y) - shift_y

        return min_x, max_x, min_y, max_y
    except (IndexError, ZeroDivisionError, cv2.error) as _:
        logger.error(
            'Ошибка при вычислении координаты углов максимального '
            'прямоугольника, вмещающего общую область.\n'
            + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def crop_intersection_area(
    img1: np.ndarray, img2: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Обрезает изображения до общей области пересечения.

    :param img1: Первое изображение.
    :param img2: Второе изображение.
    :param H: Матрица гомографии.

    :return: Обрезанные изображения.
    """
    try:
        h, w = img1.shape[:2]
        min_x, max_x, min_y, max_y = find_rect_corners(w, h, H)
        return img1[min_y:max_y, min_x:max_x], img2[min_y:max_y, min_x:max_x]
    except IndexError as _:
        logger.error(
            'Ошибка при обрезке общей области изображений.\n'
            + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def apply_intersection_area_mask(
    diff: np.ndarray, H: np.ndarray, shift_x, shift_y
) -> np.ndarray:
    """
    Применяет маску к изображению бинаризированной разницы, оставляя различия
    только на области пересечения.

    :param diff: Бинаризированное изображение разницы.
    :param H: Матрица преобразования.
    :param shift_x: Смещение по оси x.
    :param shift_y: Смещение по оси y.

    :return: Изображение разницы с примененной маской.
    """
    try:
        h, w = diff.shape[:2]
        min_x, max_x, min_y, max_y = find_rect_corners(
            w, h, H, shift_x=shift_x, shift_y=shift_y
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[min_y:max_y, min_x:max_x] = 255

        masked_diff = cv2.bitwise_and(diff, diff, mask=mask)
        return masked_diff
    except (IndexError, cv2.error) as _:
        logger.error(
            'Ошибка при применении маски общей области к изображению '
            'различий.\n' + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def match_hist(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Сопоставляет гистограмму первого изображения с гистограммой второго
    изображения.

    :param img1: Изображение, гистограмма которого будет сопоставлена.
    :param img2: Изображение, с которым будет выполнено сопоставление
    гистограммы.

    :return: Первое изображение с сопоставленной гистограммой.
    """
    try:
        hist_source = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist_reference = cv2.calcHist([img2], [0], None, [256], [0, 256])
        cdf_source = hist_source.cumsum() / float(hist_source.sum())
        cdf_reference = hist_reference.cumsum() / float(hist_reference.sum())
        lookup_table = np.interp(cdf_source, cdf_reference, np.arange(256))
        matched_img = cv2.LUT(img1, lookup_table.astype('uint8'))
        return matched_img
    except cv2.error as _:
        logger.error(
            'Ошибка при сопоставлении гистограмм.\n' + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def equalize_hist(img: np.ndarray) -> tuple[np.ndarray]:
    """
    Выравнивает гистограммы двух изображений.

    :param img: Изображение.

    :return: Изображение с выровненной гистограммой.
    """
    try:
        eq_img = cv2.equalizeHist(img)
        logger.info('Выравнивание гистограмм завершилось без ошибок')
        return eq_img
    except cv2.error as _:
        logger.error(
            'Ошибка при выравнивании гистограммы.\n' + traceback.format_exc()
        )
        raise ImageDifferenceSearcherException()


def blur(img: np.ndarray, ksize: int, sigma_x: int = 0) -> np.ndarray:
    """
    Выполняет размытие изображения.

    :param img: Изображение.
    :param ksize: Размер ядра Гауссова фильтра.
    :param sigma_x: Стандартное отклонение по оси X для Гауссова ядра.
    :return: Размытое изображение.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigma_x)


def convert_color(
    img: np.ndarray, mode: int = cv2.COLOR_BGR2GRAY
) -> np.ndarray:
    """
    Преобразует цветовое пространство двух изображений.

    :param img: Изображение.
    :param mode: Режим преобразования цветового пространства.

    :return: Изображение в новом цветовом пространстве.
    """
    try:
        cvt_img = cv2.cvtColor(img, mode)
        return cvt_img
    except cv2.error as _:
        msg = 'Ошибка при изменении цветового пространства изображения.'
        logger.error(f'{msg}\n' + traceback.format_exc())
        raise CBaseException(msg)


def resize(
    img: np.ndarray, w: int = 0, h: int = 0, fx: float = 1.0, fy: float = 1.0
) -> np.ndarray:
    """
    Изменяет размер изображения.

    :param img: Исходное изображение.
    :param w: Желаемая ширина изображения.
    :param h: Желаемая высота изображения.
    :param fx: Коэффициент масштабирования по оси x.
    :param fy: Коэффициент масштабирования по оси y.

    :return: Изображение с измененным размером.
    """
    try:
        return cv2.resize(img, (w, h), fx=fx, fy=fy)
    except cv2.error as e:
        msg = 'Ошибка при изменении размера изображения'
        logger.error(f'{msg}\n{e}\n{traceback.format_exc()}')
        raise CBaseException(msg)


def send_reg_transform_progress(
    percent_step: float, sem: Semaphore, stop_event: Event
):
    """
    Отправляет сигнал на фронтенд о том, что завершась итерация обработки
    регионов.

    :param percent_step: Значение, на которое нужно увеличить шкалу прогресса.
    :param sem: Семафора для ожидания сигнала из процессов обработки регионов.
    :param stop_event: Ивент, сообщающий, что обработка завершена
    """
    while True:
        sem.acquire()
        if stop_event.is_set():
            break


def find_reg_pair_transform(
    reg1: np.ndarray,
    reg2: np.ndarray,
    bbox: tuple,
    max_dx: int,
    max_dy: int,
    max_angle: float,
    step: float,
    thresh: float,
    max_val: float,
    sem: Semaphore,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Находит наилучшее преобразование для пары регионов изображений.

    :param reg1: Первый регион изображения.
    :param reg2: Второй регион изображения.
    :param bbox: Координаты ограничивающей рамки региона.
    :param max_dx: Максимальное смещение по оси x.
    :param max_dy: Максимальное смещение по оси y.
    :param max_angle: Максимальный угол поворота.
    :param step: Шаг угла поворота.
    :param thresh: Порог бинаризации.
    :param max_val: Максимальное значение пикселя.
    :param sem: Семафора для отправки сигнала о завершении обработки пары.

    :return: Кортеж из бинарной разницы между регионами,
    матрицы преобразования и координат ограничивающей рамки.
    """
    try:
        best_diff_area = float('inf')
        best_binary_diff = None
        best_bbox = None
        best_H = None

        for i, dx in enumerate(np.arange(-max_dx, max_dx + 1, 1)):
            for dy in np.arange(-max_dy, max_dy + 1, 1):
                for angle in np.arange(-max_angle, max_angle + 1, step):
                    cur_h, cur_w = reg2.shape[:2]
                    H = cv2.getRotationMatrix2D(
                        (cur_w / 2, cur_h / 2), angle, 1
                    )
                    H[0, 2] += dx
                    H[1, 2] += dy
                    t_reg2 = cv2.warpAffine(reg2, H, (cur_w, cur_h))
                    diff = abs_difference(reg1, t_reg2)
                    diff = apply_intersection_area_mask(diff, H, dx, dy)
                    binary_diff = binary_difference(diff, thresh, max_val)
                    diff_area = binary_diff.sum()
                    if diff_area < best_diff_area:
                        best_diff_area = diff_area
                        best_binary_diff = binary_diff
                        best_bbox = bbox
                        best_H = H
        sem.release()
        return best_binary_diff, best_H, best_bbox
    except (cv2.error, ValueError, TypeError, IndexError) as _:
        logger.error(f'Ошибка при обработке региона: {traceback.format_exc()}')
        raise ImageDifferenceSearcherException(
            'Ошибка при обработке региона. Сообщите разработчикам.'
        )


def find_best_transform(
    img1: np.ndarray,
    img2: np.ndarray,
    rows: int,
    cols: int,
    max_dx: int,
    max_dy: int,
    max_angle: float,
    step: float,
    num_process: int,
    thresh: float,
    max_val: float,
):
    """
    Находит наилучшие преобразования для каждого региона изображений.

    :param img1: Первое изображение.
    :param img2: Второе изображение.
    :param rows: Количество регионов по оси y.
    :param cols: Количество регионов по оси x.
    :param max_dx: Максимальное смещение по оси x.
    :param max_dy: Максимальное смещение по оси y.
    :param max_angle: Максимальный угол поворота.
    :param step: Шаг угла поворота.
    :param num_process: Количество процессов для выполнения вычислений.
    :param thresh: Порог бинаризации.
    :param max_val: Максимальное значение пикселя.

    :return: Список лучших регионов (кортежей из бинарной разницы,
    матрицы преобразования и ограничивающей рамки).
    """
    if not isinstance(num_process, int):
        logger.error(
            'Количество процессов в конфиге не является целым числом.'
        )
        raise ImageDifferenceSearcherException(
            'Количество процессов в конфиге должно быть целым числом.'
        )
    try:
        max_process = cpu_count()
        if num_process <= 0 or num_process > max_process:
            num_process = max_process
        h, w = img1.shape[:2]
        reg_w = w // cols
        reg_h = h // rows
        regs = []
        for i in range(cols):
            for j in range(rows):
                x1 = i * reg_w
                y1 = j * reg_h
                x2 = x1 + reg_w
                y2 = y1 + reg_h
                if i == cols - 1:
                    x2 = w - 1
                if j == rows - 1:
                    y2 = h - 1
                regs.append(
                    (img1[y1:y2, x1:x2], img2[y1:y2, x1:x2], (x1, y1, x2, y2))
                )
        percent_step = 48 / len(regs)

        with Manager() as manager:
            sem = manager.Semaphore(0)
            stop_event = manager.Event()
            t = Thread(
                target=send_reg_transform_progress,
                args=(percent_step, sem, stop_event),
            )
            t.start()
            with Pool(processes=num_process) as p:
                best_regions = p.starmap_async(
                    partial(
                        find_reg_pair_transform,
                        max_dx=max_dx,
                        max_dy=max_dy,
                        max_angle=max_angle,
                        step=step,
                        thresh=thresh,
                        max_val=max_val,
                        sem=sem,
                    ),
                    regs,
                )
                best_regions.wait()
            stop_event.set()
            sem.release()
            t.join()
        return best_regions.get()
    except (IndexError, ZeroDivisionError) as _:
        logger.error(
            f'Некорректно заданное количество регионов или непредвиденная '
            f'ошибка. {traceback.format_exc()}'
        )
        raise ImageDifferenceSearcherException(
            'Ошибка при дополнительной деформации регионов. Убедитесь, что '
            'cols и rows в конфиге заданны корректно.'
        )
    except Exception as _:
        logger.error(
            f'Ошибка с процессами и/или потоками. {traceback.format_exc()}'
        )
        raise ImageDifferenceSearcherException(
            'Непредвиденная ошибка при обработке регионов. '
            'Сообщите данные разработчикам.'
        )


def get_file_path(path):
    """Загружает изображение из ресурсов приложения."""
    try:
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        if getattr(sys, '_MEIPASS', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        file_path = os.path.join(base_path, directory, filename)
        return file_path
    except Exception as _:
        msg = 'Ошибка при получении статического файла.'
        logger.error(f'{msg}\n' + traceback.format_exc())
        raise CBaseException(msg)
