import numpy as np
import torch

import tools
from exceptions import FeatureMatcherException


class SuperPointNet(torch.nn.Module):
    """Определение сети SuperPoint в Pytorch."""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Общий энкодер.

        self.conv1a = torch.nn.Conv2d(
            1, c1, kernel_size=3, stride=1, padding=1
        )
        self.conv1b = torch.nn.Conv2d(
            c1, c1, kernel_size=3, stride=1, padding=1
        )
        self.conv2a = torch.nn.Conv2d(
            c1, c2, kernel_size=3, stride=1, padding=1
        )
        self.conv2b = torch.nn.Conv2d(
            c2, c2, kernel_size=3, stride=1, padding=1
        )
        self.conv3a = torch.nn.Conv2d(
            c2, c3, kernel_size=3, stride=1, padding=1
        )
        self.conv3b = torch.nn.Conv2d(
            c3, c3, kernel_size=3, stride=1, padding=1
        )
        self.conv4a = torch.nn.Conv2d(
            c3, c4, kernel_size=3, stride=1, padding=1
        )
        self.conv4b = torch.nn.Conv2d(
            c4, c4, kernel_size=3, stride=1, padding=1
        )
        # Голова детектора.

        self.convPa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1
        )
        self.convPb = torch.nn.Conv2d(
            c5, 65, kernel_size=1, stride=1, padding=0
        )
        # Голова дескриптора.

        self.convDa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1
        )
        self.convDb = torch.nn.Conv2d(
            c5, d1, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход, который совместно вычисляет необработанные тензоры точек
        и дескрипторов.

        :param x: Входной тензор изображения Pytorch формы N x 1 x H x W.
        :return: Кортеж из двух тензоров:
            - Выходной тензор точек формы N x 65 x H/8 x W/8.
            - Выходной тензор дескрипторов формы N x 256 x H/8 x W/8.
        """
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc


class SuperPoint:
    """Обертка вокруг сети Pytorch, помогающая с предварительной
    и последующей обработкой изображений."""

    def __init__(
        self,
        weights_path: str,
        nms_dist: int,
        conf_thresh: float,
        nn_thresh: float,
    ):
        """
        Инициализация объекта SuperPoint.

        :param weights_path: Путь к файлу с весами модели.
        :param nms_dist: Расстояние для подавления немаксимумов.
        :param conf_thresh: Порог достоверности для точек.
        :param nn_thresh: Порог для ближайшего соседа.
        :param cuda: Использовать ли CUDA (если доступно).
        """
        self.name = 'SuperPoint'
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell = 8
        self.border_remove = 4

        self.net = SuperPointNet()
        weights_path = tools.get_file_path(weights_path)
        self.net.load_state_dict(
            torch.load(
                weights_path, map_location=lambda storage, loc: storage
            )
        )
        self.net.eval()

    def nms_fast(
        self, in_corners: np.ndarray, H: int, W: int, dist_thresh: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Приблизительное подавление немаксимумов на углах numpy формы:
          3xN [x_i,y_i,conf_i]^T

        Краткое описание алгоритма: создайте сетку размером HxW. Присвойте
        каждому угловому местоположению 1, остальные - нули.
        Пройдите по всем единицам и преобразуйте их в -1 или 0.
        Подавите точки, установив близлежащие значения в 0.

        Условные обозначения значений сетки:
        -1: Сохранено.
         0: Пусто или подавлено.
         1: Подлежит обработке (преобразуется в сохраненное или подавленное).

        ПРИМЕЧАНИЕ: NMS сначала округляет точки до целых чисел, поэтому
        расстояние NMS может не точно соответствовать dist_thresh.
        Также предполагается, что точки находятся в пределах границ
        изображения.

        :param in_corners: Массив numpy 3xN с углами
        [x_i, y_i, confidence_i]^T.
        :param H: Высота изображения.
        :param W: Ширина изображения.
        :param dist_thresh: Расстояние для подавления, измеряемое
        как расстояние бесконечной нормы.
        :return: Кортеж из двух массивов numpy:
            - nmsed_corners: Матрица numpy 3xN с выжившими углами.
            - nmsed_inds: Вектор numpy длины N с индексами выживших углов.
        """
        grid = np.zeros((H, W)).astype(int)
        inds = np.zeros((H, W)).astype(int)
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Округленные углы.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        count = 0
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:
                grid[
                    pt[1] - pad:pt[1] + pad + 1,
                    pt[0] - pad:pt[0] + pad + 1,
                ] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def detect_and_compute(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Извлечение особых точек и дескрипторов.

        :param img: Входное изображение HxW numpy float32 в диапазоне [0,1].
        :return: Кортеж из трех элементов:
            - Углы: массив numpy 3xN с углами [x_i, y_i, confidence_i]^T.
            - Дескрипторы: массив numpy 256xN соответствующих
            нормализованных дескрипторов единиц.
            - Карта координат: тепловая карта HxW numpy в диапазоне [0,1]
            достоверности точек.
        """
        try:
            H, W = img.shape[0], img.shape[1]
            inp = img.copy()
            inp = inp.reshape(1, H, W)
            inp = torch.from_numpy(inp)
            inp = torch.autograd.Variable(inp).view(1, 1, H, W)
            # Прямой проход сети.

            outs = self.net.forward(inp)
            semi, coarse_desc = outs[0], outs[1]

            semi = semi.data.cpu().numpy().squeeze()

            # --- Обработка точек.

            dense = np.exp(semi)
            dense = dense / (np.sum(dense, axis=0) + 0.00001)
            nodust = dense[:-1, :, :]
            Hc = int(H / self.cell)
            Wc = int(W / self.cell)
            nodust = nodust.transpose(1, 2, 0)
            heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
            heatmap = np.transpose(heatmap, [0, 2, 1, 3])
            heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
            xs, ys = np.where(
                heatmap >= self.conf_thresh
            )  # Порог достоверности.
            if len(xs) == 0:
                return np.zeros((3, 0)), None, None
            pts = np.zeros(
                (3, len(xs))
            )  # Заполнение данных точки размером 3xN.
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = self.nms_fast(
                pts, H, W, dist_thresh=self.nms_dist
            )  # Применение NMS.
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Сортировка по достоверности.
            # Удаление точек вдоль границы.

            bord = self.border_remove
            toremoveW = np.logical_or(
                pts[0, :] < bord, pts[0, :] >= (W - bord)
            )
            toremoveH = np.logical_or(
                pts[1, :] < bord, pts[1, :] >= (H - bord)
            )
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            # --- Обработка дескриптора.

            D = coarse_desc.shape[1]
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                # Интерполяция в карту дескрипторов с использованием
                # местоположений 2D-точек.

                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.0)) - 1.0
                samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.0)) - 1.0
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            return pts, desc, heatmap
        except Exception as _:
            raise FeatureMatcherException()


if __name__ == '__main__':
    model = SuperPointNet()
    model.load_state_dict(
        torch.load('superpoint_v1.pth', map_location=torch.device('cpu')))
    model.eval()

    traced_model = torch.jit.trace(model, torch.rand(1, 1, 640, 480))
    traced_model.save('superpoint_v3.pt')
    # torch.save(model.state_dict(), 'superpoint_v3.pt')
