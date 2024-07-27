import cv2
import numpy as np

# Параметры для Farneback
flow_params = dict(pyr_scale=0.5,  # Уменьшение изображения на каждом уровне пирамиды
                   levels=3,  # Количество уровней пирамиды
                   winsize=15,  # Размер окна для поиска соответствий
                   iterations=3,  # Количество итераций алгоритма
                   poly_n=5,  # Размер полинома для аппроксимации окна
                   poly_sigma=1.2,  # Стандартное отклонение Гауссова ядра
                   flags=0)  # Флаги алгоритма


def select_roi(frame):
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Frame")
    return bbox


# Загрузка видеопотока
video = cv2.VideoCapture('test_movies//popadanie.mpg')
if not video.isOpened():
    print("Не удалось открыть видеопоток")
    exit()

ret, prev_frame = video.read()
if not ret:
    print("Не удалось получить кадр")
    exit()

bbox_pts = None
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


def newPointWC(event, x, y, flags, param):
    """
    Обработка события OpenCV
    :param event: Событие
    :param x: координаты OX
    :param y: координаты OY
    :param flags: не используется
    :param param: не используется
    """
    global bbox_pts, wait
    if event == cv2.EVENT_RBUTTONDOWN:
        wait = 1
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32).reshape(-1, 1, 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        wait = 1
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32).reshape(-1, 1, 2)


cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Tracking", newPointWC)


wait = 27

while True:
    # Получение текущего кадра
    ret, frame = video.read()
    if not ret:
        break

    frame = frame[:, :]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if bbox_pts is not None:
        # Расчет оптического потока (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)

        # Вычисление средней скорости потока внутри ROI
        x, y, w, h = cv2.boundingRect(bbox_pts.astype(np.int32))
        roi_flow = flow[y:y + h, x:x + w]
        mean_flow = np.mean(roi_flow, axis=(0, 1))

        # Обновление ROI на основе средней скорости
        bbox_pts[:, :, 0] += mean_flow[0]
        bbox_pts[:, :, 1] += mean_flow[1]

        # Отрисовка ROI
        transformed_bbox_int = np.int32(bbox_pts)
        cv2.polylines(frame, [transformed_bbox_int], True, (0, 255, 0), 2)

    # Обновление предыдущего кадра
    prev_gray = gray.copy()

    # Отображение результата
    cv2.imshow("Tracking", frame)

    # Выход по нажатию клавиши 'q'
    print(wait)
    if cv2.waitKey(wait) & 0xFF == ord('q'):
        break
    if cv2.waitKey(wait) & 0xFF == ord('s'):
        bbox_pts = None
        wait = 27


# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()