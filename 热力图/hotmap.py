import cv2
import numpy as np
import math

# 定义高斯分布函数
def gaussian(x, y, sigma):
    return math.exp(-(x**2 + y**2)/(2*sigma**2))

# 定义绘制高斯分布函数
def draw_gaussian(img, center, sigma, color):
    x, y = center
    height, width = img.shape[0], img.shape[1]
    for i in range(height):
        for j in range(width):
            dist = math.sqrt((i-y)**2 + (j-x)**2)
            img[i,j] += color * gaussian(dist, 0, sigma)

# 定义图像大小和高斯分布参数
width, height = 512, 512
sigma = 50

# 创建一个空白图像
img = np.zeros((height, width), dtype=np.uint8)

# 在图像中心添加高斯分布
center = (int(width/2), int(height/2))
draw_gaussian(img, center, sigma, 255)

# 将图像转换为彩色图像，并根据像素值着色
img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# 显示结果
cv2.imshow("Gaussian Distribution", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()