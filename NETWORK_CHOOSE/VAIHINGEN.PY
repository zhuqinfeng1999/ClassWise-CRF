import numpy as np


def select_networks(iou_matrix, k, methods):
    """
    选择k个网络，使得融合后的mIoU最高。

    参数：
    iou_matrix: n x c 的numpy数组，n个网络在c个类别上的IoU。
    k: 选择的网络数量。
    methods: 网络名称列表，与iou_matrix的行对应。

    返回：
    选择的网络名称列表和对应的mIoU。
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

    # 计算最终的mIoU
    final_max_iou = np.max(iou_matrix[selected_indices], axis=0)
    final_miou = np.mean(final_max_iou)
    return [methods[i] for i in selected_indices], final_miou


# Vaihingen数据集的网络和IoU表现
methods = [
    "segformerb0", "upernetr50", "pspnetr50", "deeplabv3pr50",
    "bisenetv1r50", "convnextt", "vmambat", "swint"
]
iou_matrix = np.array([
    [81.46, 88.24, 66.25, 81.18, 62.51, 50.72],
    [80.86, 88.59, 64.31, 80.05, 52.63, 50.26],
    [81.01, 88, 63.81, 79.4, 48.46, 50.01],
    [80.19, 87.53, 63.23, 79.35, 51.35, 50.7],
    [81.5, 89.04, 65.99, 80.41, 55.65, 50.75],
    [83.35, 90.65, 68.75, 82.63, 72.57, 49.68],
    [83.39, 91.11, 68.91, 83.04, 70.46, 49.51],
    [84, 91.16, 68.55, 82.67, 72.03, 50.36]
])

# 选择2个网络
selected_methods_k2, miou_k2 = select_networks(iou_matrix, 2, methods)
print("选择2个网络的结果:", selected_methods_k2)
print("融合后的mIoU:", round(miou_k2, 2))

# 选择3个网络
selected_methods_k3, miou_k3 = select_networks(iou_matrix, 3, methods)
print("选择3个网络的结果:", selected_methods_k3)
print("融合后的mIoU:", round(miou_k3, 2))