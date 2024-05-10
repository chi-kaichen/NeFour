import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.figure(dpi=2000)

a_path = '1.png'
b_path = '2.png'
c_path = '3.png'
d_path = '4.png'

# 读取图像数据并转换为浮点数数组
A = plt.imread(a_path).astype(np.float32) / 255
B = plt.imread(b_path).astype(np.float32) / 255
C = plt.imread(c_path).astype(np.float32) / 255
D = plt.imread(d_path).astype(np.float32) / 255

A_flat = A.reshape(A.shape[0], -1)
B_flat = B.reshape(B.shape[0], -1)
C_flat = C.reshape(C.shape[0], -1)
D_flat = D.reshape(D.shape[0], -1)

n_components = 2  # 降维后的维度
tsne = TSNE(n_components=n_components, random_state=90)

A_tsne = tsne.fit_transform(A_flat)
B_tsne = tsne.fit_transform(B_flat)
C_tsne = tsne.fit_transform(C_flat)
D_tsne = tsne.fit_transform(D_flat)
plt.scatter(A_tsne[:, 0], A_tsne[:, 1], c='red', marker='x', label='A')
plt.scatter(B_tsne[:, 0], B_tsne[:, 1], c='blue', marker='^', label='B')
plt.scatter(C_tsne[:, 0], C_tsne[:, 1], c='green', marker='o', label='C')
plt.scatter(D_tsne[:, 0], D_tsne[:, 1], c='orange', marker='v', label='D')
plt.legend()
plt.show()