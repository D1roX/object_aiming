import cv2
import numpy as np

akaze = cv2.ORB_create()


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


def newPointWC(event, x, y, flags, param):
    """
        Обработка события OpenCV
        :param event: Событие
        :param x: координаты OX
        :param y: координаты OY
        :param flags: не используется
        :param param: не используется
    """
    global bbox_pts
    if event == cv2.EVENT_RBUTTONDOWN:
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32).reshape(-1, 1, 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        x, y, w, h = [x - 30, y - 30, 60, 60]
        bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32).reshape(-1, 1, 2)


# bbox = select_roi(prev_frame.copy())
# x, y, w, h = bbox

# bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
#                     dtype=np.float32).reshape(-1, 1, 2)

cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Tracking", newPointWC)

while True:
    # Получение текущего кадра
    ret, frame = video.read()
    if not ret:
        break

    frame = frame[400:, 600:]

    # Преобразование кадров в градации серого
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Поиск ключевых точек и дескрипторов
    kp1, des1 = akaze.detectAndCompute(gray_prev_frame, None)
    kp2, des2 = akaze.detectAndCompute(gray_frame, None)

    # Сопоставление дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # flann_index_params = {
    #     'algorithm': 1,
    #     'trees': 5
    # }
    # flann_search_params = {
    #     'checks': 100
    # }
    #
    # bf = cv2.FlannBasedMatcher(
    #         flann_index_params, flann_search_params
    #     )

    # des1 = np.float32(des1)
    # des2 = np.float32(des2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Применение Lowe's ratio теста
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # Поиск гомографии между кадрами
    if len(good) > 10 and bbox_pts is not None:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1,
                                                                         2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1,
                                                                         2)
        M, _ = cv2.findHomography(
            kp1, kp2, cv2.RANSAC, 10.0, confidence=0.99, maxIters=2000
        )

        # Трансформация точек рамки
        print(bbox_pts)
        transformed_bbox = cv2.perspectiveTransform(bbox_pts, M)

        transformed_bbox_int = np.int32(transformed_bbox)
        cv2.rectangle(frame, tuple(transformed_bbox_int[0, 0]),
                      tuple(transformed_bbox_int[2, 0]), (0, 255, 0), 2)

        # Обновление точек рамки для следующей итерации
        bbox_pts = transformed_bbox

    # Отображение результата
    cv2.imshow("Tracking", frame)

    # Обновление предыдущего кадра
    prev_frame = frame.copy()

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()
