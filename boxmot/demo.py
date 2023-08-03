from boxmot import DeepOCSORT
from pathlib import Path
import cv2
import numpy as np
import yolov5

tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.onnx"),  # 加载重识别模型
    device="cuda:0",  # GPU设备号
    fp16=True,  # 计算精度
)  # 定义跟踪器

model = yolov5.load("yolov5x.pt")  # 定义检测器，加载与训练模型

trackHistory = []  # 用栈保存跟踪历史
historyLength = 2000  # 跟踪历史长度
CacheLength = 200  # 跟踪历史缓存长度

# 待入侵检测区域
ImageROI = np.array(
    [
        [0, 0, 100, 100],
        [300, 300, 600, 600],
    ]
)
# 假定最大目标id
MaxId = 100
# 入侵帧数阈值
IntrusionThreshold = 150
# 用于画出ROI区域的图像
zeroImg = np.zeros(np.array([720, 1280, 3]), dtype=np.uint8)

vid = cv2.VideoCapture("ShakeFloor.mp4")  # 用视频模拟图像流输入
color = (0, 0, 255)  # BGR
thickness = 2  # 演示图像参数
fontscale = 0.5  # 演示图像参数


def track(dets, im):
    # dets 为检测结果 --> (left, up, right, bottom, conf, cls)
    # im 为图像 --> numpy数组，unit8类型
    ts = tracker.update(
        dets, im
    )  # ts为跟踪结果 --> (left, up, right, bottom, id, conf, cls)
    if ts.size == 0:  # 无跟踪结果（首帧，无检测结果等情况触发）
        return np.empty((0, 5))  # 返回空数组
    bbox = ts[:, 0:4].astype("int")  # float64 to int
    id = ts[:, 4].astype("int")  # float64 to int
    conf = ts[:, 5]  # 置信度
    cls = ts[:, 6]  # 类别
    bottom_center = np.array(
        [(bbox[:, 0] + bbox[:, 2]) / 2, bbox[:, 3]]
    ).T  # 计算目标下边界框中心
    result = np.concatenate(
        (bottom_center, id[:, None], conf[:, None], cls[:, None]), 1
    )  # result 为最终结果 --> (bottom_center_x, bottom_center_y, id, conf, cls)
    return result


# 进行入侵检测，返回检测结果，以下中心为依据


# 截取历史记录
def getHistory(trackHistory, historyLength):
    if len(trackHistory) > historyLength:
        return trackHistory[-historyLength:]
    else:
        return trackHistory

def checkIntrusion(trackHistory, ImageROI, IntrusionThreshold, MaxId=100):
    IntrusionCount = np.zeros(MaxId, dtype="int")  # 创建计数器
    # # 检测目标是否入侵ROI区域，记录跟踪历史中目标入侵次数
    for i in range(len(trackHistory)):
        for j in range(len(trackHistory[i])):
            for roi in ImageROI:
                if (
                    trackHistory[i][j][0] > roi[0]
                    and trackHistory[i][j][0] < roi[2]
                    and trackHistory[i][j][1] > roi[1]
                    and trackHistory[i][j][1] < roi[3]
                ):
                    IntrusionCount[int(trackHistory[i][j][2])] += 1
    # 获取当前跟踪结果，并在最后一列添加入侵状态信息
    result = np.append(trackHistory[-1], np.zeros((len(trackHistory[-1]), 1)), 1)
    # 查询状态，在result中添加是否入侵的状态信息
    for i in range(len(result)):
        if IntrusionCount[int(result[i][2])] > IntrusionThreshold:
            result[i][-1] = 1  # 确认入侵
        else:
            result[i][-1] = 0  # 未入侵

    return result


# 下面的代码为整体流程模拟

while True:  # 模拟图像流输入
    ret, im = vid.read()  # 获取单帧图像

    if im is None:
        break

    results = model(im)  # 进行检测推理

    dets = results.xyxy[0].cpu().numpy()  # 提取检测结果 dets --> (x, y, x, y, conf, cls)

    result = track(
        dets, im
    )  # 获取跟踪结果 result--> (bottom_center_x, bottom_center_y, id, conf, cls)

    trackHistory = trackHistory[-historyLength:] + [result]  # 保存跟踪历史

    # 截取帧跟踪历史

    cacheHistory = getHistory(trackHistory, CacheLength)

    checkResult = checkIntrusion(
        cacheHistory, ImageROI, IntrusionThreshold, MaxId=MaxId
    )  # 获取入侵检测结果

    if result.size == 0:  # 无跟踪结果直接跳过
        continue

    # 对结果进行可视化

    # 画出ROI区域，内部覆盖浅红色
    for roi in ImageROI:
        im = cv2.rectangle(im, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
        im = im + cv2.rectangle(
            zeroImg, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 30), -1
        )

    if result.shape[0] != 0:
        for re, cre in zip(result, checkResult):
            im = cv2.circle(im, (int(re[0]), int(re[1])), 5, (0, 255, 0), -1)
            # 画出目标检测框下中心

            cv2.putText(
                im,
                f"id: {re[2]:.0f}, conf: {re[3]:.2f}, c: {re[4]:.0f}, i: {cre[5]:.0f}",
                (int(re[0]), int(re[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
            )  # 标示id conf class

    # show image
    cv2.imshow("frame", im)

    # 推出按键
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
