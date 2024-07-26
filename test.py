import cv2
import numpy as np

# Инициализация AKAZE детектора и дескриптора
akaze = cv2.AKAZE_create()

# Функция для выделения области с помощью мыши
def select_roi(frame):
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Frame")
    return bbox

# Загрузка видеопотока
video = cv2.VideoCapture('test_movies//2_2x_crop.mp4')
if not video.isOpened():
    print("Не удалось открыть видеопоток")
    exit()

# Получение первого кадра
ret, prev_frame = video.read()
if not ret:
    print("Не удалось получить кадр")
    exit()

# Выделение объекта на первом кадре
bbox = select_roi(prev_frame.copy())
x, y, w, h = bbox

# Инициализация точек рамки
bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(-1, 1, 2)

while True:
    # Получение текущего кадра
    ret, frame = video.read()
    if not ret:
        break

    # Преобразование кадров в градации серого
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Поиск ключевых точек и дескрипторов
    kp1, des1 = akaze.detectAndCompute(gray_prev_frame, None)
    kp2, des2 = akaze.detectAndCompute(gray_frame, None)

    # Сопоставление дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Применение Lowe's ratio теста
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Поиск гомографии между кадрами
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Трансформация точек рамки
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()