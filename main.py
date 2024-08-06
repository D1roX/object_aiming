import cv2
import numpy as np
import time

# Задаем пути к файлам
VIDEO_PATH = "D:/VSCodeProjects/object_aim/test_movies/FL-10.mp4"
SCOUT_IMG_PATH = "D:/VSCodeProjects/object_aim/test_imgs/search1.jpg"
SKIP_FRAME = 250
RATIO_THRESH = 0.75

class ObjectAim:
    def __init__(self):
        # Загружаем scout-изображение
        self.scout_img = cv2.imread(SCOUT_IMG_PATH)
        if self.scout_img is None:
            print("Не удалось загрузить scout-изображение")
            exit(1)

        # Инициализируем видеозахват
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            print("Не удалось открыть видео")
            exit(1)

        # Инициализируем детектор и сопоставитель
        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.scout_kpts, self.scout_des = self.detector.detectAndCompute(self.scout_img, None)

        # Выбираем область интереса (ROI)
        self.bbox = cv2.selectROI("Select ROI", self.scout_img)
        cv2.destroyAllWindows()
        print("Bounding box:", self.bbox)

        # Инициализируем текущую область интереса
        self.cur_bbox = self.bbox

    def run(self):
        frame_count = 0
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count < SKIP_FRAME:
                frame_count += 1
                continue

            # Отрисовка текущего bbox на кадре
            cv2.rectangle(frame, self.cur_bbox, (0, 255, 0), 2)

            # Сопоставление ключевых точек
            frame_kpts, frame_des = self.detector.detectAndCompute(frame, None)
            matches = self.matcher.match(self.scout_des, frame_des)

            # Фильтрация по порогу отношения
            good_matches = [match for match in matches if match.distance < RATIO_THRESH * matches[0].distance]

            img_matches = np.zeros((
                self.scout_img.shape[0] + frame.shape[0],
                self.scout_img.shape[1] + frame.shape[1],
                3
            ), dtype=np.uint8)
            img_matches = cv2.drawMatches(
                self.scout_img, self.scout_kpts, frame, frame_kpts,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            if len(good_matches) < 4:
                cv2.imshow("Matches", img_matches)
                cv2.waitKey(1)
                print("No good matches found!")
                continue

            # Вычисление гомографии
            scout_pts = np.float32([self.scout_kpts[match.queryIdx].pt for match in good_matches])
            frame_pts = np.float32([frame_kpts[match.trainIdx].pt for match in good_matches])
            H, mask = cv2.findHomography(scout_pts, frame_pts, cv2.RANSAC)

            # Преобразование bbox
            bbox_points = np.float32([
                [self.bbox[0], self.bbox[1]],
                [self.bbox[0] + self.bbox[2], self.bbox[1]],
                [self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]],
                [self.bbox[0], self.bbox[1] + self.bbox[3]]
            ])
            bbox_points_mat = cv2.perspectiveTransform(bbox_points.reshape(1, -1, 2), H)

            self.cur_bbox = cv2.boundingRect(bbox_points_mat.reshape(-1, 2))
            self.cur_bbox = (self.cur_bbox[0] + self.scout_img.shape[1], self.cur_bbox[1] + self.scout_img.shape[0], self.cur_bbox[2], self.cur_bbox[3])

            # Отрисовка bbox на кадре
            cv2.rectangle(img_matches, self.cur_bbox, (0, 255, 0), 2)

            cv2.imshow("Matches", img_matches)
            cv2.waitKey(1)

            end_time = time.time()
            print(f"Время выполнения: {end_time - start_time:.3f} секунд")

if __name__ == "__main__":
    start_time = time.time()
    object_aim = ObjectAim()
    object_aim.run()
    end_time = time.time()
    print(f"Общее время выполнения: {end_time - start_time:.3f} секунд")