import numpy as np
import time
import itertools


def select_networks_greedy(iou_matrix, k):
    """
    使用贪婪算法选择k个网络，使得融合后的mIoU最高。
    参数：
    iou_matrix: n x c 的numpy数组，n个网络在c个类别上的IoU。
    k: 选择的网络数量。
    返回：
    选择的网络索引列表。
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
    return selected_indices


def select_networks_exhaustive(iou_matrix, k):
    """
    使用遍历算法（穷举）选择k个网络，使得融合后的mIoU最高。
    参数：
    iou_matrix: n x c 的numpy数组，n个网络在c个类别上的IoU。
    k: 选择的网络数量。
    返回：
    选择的网络索引列表。
    """
    n, c = iou_matrix.shape
    best_miou = -1
    best_comb = None
    for comb in itertools.combinations(range(n), k):
        max_iou_per_class = np.max(iou_matrix[list(comb)], axis=0)
        miou = np.mean(max_iou_per_class)
        if miou > best_miou:
            best_miou = miou
            best_comb = comb
    return list(best_comb)


# 测试函数，测量时间
def test_selection_time(n, c, k_values):
    # 随机生成IoU矩阵，值在0-100之间
    iou_matrix = np.random.uniform(0, 100, size=(n, c))

    results = {}
    for k in k_values:
        # 贪婪算法时间
        start_time = time.time()
        greedy_indices = select_networks_greedy(iou_matrix, k)
        greedy_time = time.time() - start_time

        # 遍历算法时间
        start_time = time.time()
        exhaustive_indices = select_networks_exhaustive(iou_matrix, k)
        exhaustive_time = time.time() - start_time

        results[k] = {
            'greedy_time': greedy_time,
            'exhaustive_time': exhaustive_time
        }
    return results


# 参数设置
c = 7  # 假设7个类别，如LoveDA数据集
k_values = [2, 3, 4]

# 测试N=10
print("Testing for N=10")
results_n10 = test_selection_time(10, c, k_values)
for k, times in results_n10.items():
    print(f"K={k}: Greedy time = {times['greedy_time']:.4f}s, Exhaustive time = {times['exhaustive_time']:.4f}s")

# 测试N=100 (K=4可能很慢，视机器性能而定)
print("\nTesting for N=100")
results_n100 = test_selection_time(100, c, k_values)
for k, times in results_n100.items():
    print(f"K={k}: Greedy time = {times['greedy_time']:.4f}s, Exhaustive time = {times['exhaustive_time']:.4f}s")