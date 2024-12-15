import numpy as np

# 假设 Y[node] 已经定义好
Y_node = np.array([0, 0, 1, 0, 0, 0, 0])  # 示例 ndarray，实际应为你提供的数据

# 使用 np.argmax 找到值为 1 的索引
index_of_one = np.argmax(Y_node)

print(index_of_one)
