import cv2
import os

# 设置视频编解码器和帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0

# 获取文件夹中的图像文件名并按名称排序
path = '/home/liumu/code/yolo_tracking/demoResult/Tra/'
img_names = os.listdir(path)
img_names.sort()

# 获取第一张图像的大小并创建视频写入对象
img_path = os.path.join(path, img_names[0])
img = cv2.imread(img_path)
height, width, channels = img.shape
video_path = 'tra.mp4'
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# 逐一读取图像并写入视频
for img_name in img_names:
    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)
    video_writer.write(img)
    print(img_name)

# 释放视频写入对象
video_writer.release()