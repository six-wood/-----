from boxmot import DeepOCSORT
from pathlib import Path
import cv2
import numpy as np
import yolov5

tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.onnx"),  # 加载重识别模型
    device="cuda:0", # GPU设备号
    fp16=True, #计算精度
) # 定义跟踪器

model = yolov5.load("yolov5x.pt") # 定义检测器，加载与训练模型

vid = cv2.VideoCapture("ShakeFloor.mp4") # 用视频模拟图像流输入
color = (0, 0, 255)  # BGR
thickness = 2 #演示图像参数
fontscale = 0.5 # 演示图像参数


def track(dets, im): 
    # dets 为检测结果 --> (left, up, right, bottom, conf, cls)
    # im 为图像 --> numpy数组，unit8类型
    ts = tracker.update(dets, im)  # ts为跟踪结果 --> (left, up, right, bottom, id, conf, cls)
    if ts.size == 0: # 无跟踪结果（首帧，无检测结果等情况触发）
        return np.empty((0, 5)) # 返回空数组
    bbox = ts[:, 0:4].astype("int")  # float64 to int
    id = ts[:, 4].astype("int")  # float64 to int
    conf = ts[:, 5] # 置信度
    cls = ts[:, 6] # 类别
    bottom_center = np.array(
        [(bbox[:, 0] + bbox[:, 2]) / 2, bbox[:, 3]]
    ).T  # 计算目标下边界框中心
    result = np.concatenate(
        (bottom_center, id[:, None], conf[:, None], cls[:, None]), 1
    ) # result 为最终结果 --> (bottom_center_x, bottom_center_y, id, conf, cls)
    return result

# 下面的代码为整体流程模拟

while True: # 模拟图像流输入
    ret, im = vid.read() #获取单帧图像

    if im is None:
        break

    results = model(im)  # 进行检测推理

    dets = results.xyxy[0].cpu().numpy()  # 提取检测结果 dets --> (x, y, x, y, conf, cls)
    dd = np.empty((0, 6))
    for det in dets:
        if det[5] == 0:
            dd = np.vstack((dd, det)) # 类别筛选

    result = track(dd, im)  # 获取跟踪结果 result--> (bottom_center_x, bottom_center_y, id, conf, cls)

    if result.size == 0: # 无跟踪结果直接跳过
        continue

    # 对结果进行可视化
    if result.shape[0] != 0:
        for re in result:

            im = cv2.circle(im, (int(re[0]), int(re[1])), 5, (0, 255, 0), -1)
            # 画出目标检测框下中心

            cv2.putText(
                im,
                f"id: {re[2]}, conf: {re[3]}, c: {re[4]}",
                (int(re[0]), int(re[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
            )# 标示id conf class

    # show image
    cv2.imshow("frame", im)

    # 推出按键
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
