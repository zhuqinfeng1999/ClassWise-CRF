import numpy as np


def select_networks(iou_matrix, k, methods):
    """
    选择k个网络，使得融合后的mIoU最高。

    参数：
    iou_matrix: n x c 的numpy数组，n个网络在c个类别上的IoU。
    k: 选择的网络数量。
    methods: 网络名称列表，与iou_matrix的行对应。

    返回：
    选择的网络名称列表。
    """
    n, c = iou_matrix.shape
    selected_indices = []

    for _ in range(k):
        best_miou = -1
        best_net = -1
        for i in range(n):
            if i in selected_indices:
                continue
            # 计算加入网络i后的mIoU
            current_selected = selected_indices + [i]
            max_iou_per_class = np.max(iou_matrix[current_selected], axis=0)
            miou = np.mean(max_iou_per_class)
            if miou > best_miou:
                best_miou = miou
                best_net = i
        selected_indices.append(best_net)

    # 返回网络名称
    return [methods[i] for i in selected_indices]


# 输入数据
methods = [
    "segformerb0", "upernetr50", "pspnetr50", "deeplabv3pr50",
    "bisenetv1r50", "convnextt", "vmambat", "swint"
]
iou_matrix = np.array([
    [53.23, 62.71, 52.12, 63.4, 29.37, 41.93, 45.65],
    [54.04, 60.14, 50.21, 59.86, 25, 40.84, 50.24],
    [53.85, 60.57, 51.37, 58.76, 21.92, 42.31, 50.41],
    [53.44, 61.63, 53.5, 62.3, 26.02, 42.69, 49.26],
    [53.31, 55.97, 50.58, 59.85, 27.75, 40.15, 46.74],
    [55.58, 66.08, 56.32, 68.61, 32.82, 42.33, 55.88],
    [55.29, 65.08, 56.89, 71.31, 33.68, 40.17, 53.52],
    [54.61, 65.37, 55.98, 69.79, 31.69, 44.14, 52.19]
])

# 选择3个网络
selected_methods_k3 = select_networks(iou_matrix, 3, methods)
print("选择3个网络的结果:", selected_methods_k3)


# 选择2个网络
selected_methods_k2 = select_networks(iou_matrix, 2, methods)
print("选择2个网络的结果:", selected_methods_k2)