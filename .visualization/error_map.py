import cv2
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
plt.figure(dpi=4000)
image1 = cv2.imread('610ref.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('610gl.png', cv2.IMREAD_GRAYSCALE)

error_map = cv2.absdiff(image1, image2)
#normalized_error_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#cmap_list = ['blue', 'cyan', 'green', 'magenta', 'red']
#custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_list, N=5)
# 显示热力图
plt.imshow(error_map, cmap = 'CMRmap')
plt.colorbar()
plt.show()