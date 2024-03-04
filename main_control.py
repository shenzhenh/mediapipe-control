import csv
import copy
import argparse
import itertools
import time
import pyautogui
import pydirectinput
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import FpsCal
from model import GestureClassifier

pydirectinput.PAUSE = 0.01


# 得到手部的外接矩形的位置
def calc_bounding_rect(image, landmarks):
    # 获取图像的宽,高
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    # 由于特征点的位置坐标被归一化为[0,1]区间，这里恢复为真正的相对位置
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    # 使用boundingRect函数
    # 通过这些特征点的点集,找到最小外接矩形的位置，也就是手部的位置
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


#   获取特征点集的列表，都是具体的位置
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # 通过遍历得到每个关键点在图像上的具体位置
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


# 对得到的手部关键点位置坐标数据，进行预处理操作
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # 转换为相对的坐标
    base_x, base_y = 0, 0
    # 通过遍历,得到其余关键点相对于0号关键点(即手腕处)的相对位置
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # 转化为一个一维的列表
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    # 标准化
    # 首先得到最大值
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    # 所有数据除以最大值,得到标准化后的数据
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # 返回该列表
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 对数据进行标准化
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # 转化为一维列表
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


# 调用draw_landmarks函数,画出特征点的位置以及连线
def my_draw(image, hand_landmaks):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmaks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())
    return image


# 在图像上画矩形框,本程序中是圈出手的位置
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (232, 162, 239), 1)
    return image


# 在图像上画文本框
def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    # 在手部识别框上面画一个矩形作为文本框,并设置好底色
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (18, 77, 104), -1)
    # 文本框中写入相关信息,包括识别出的左右手,以及手的姿势
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


# 在图像上作出帧率信息
def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (450, 30), cv.FONT_HERSHEY_TRIPLEX,
               0.8, (139, 249, 194), 1, cv.LINE_AA)

    # 如果选择了录入信息模式,则也会给出显示
    mode_string = ['Logging Key Point']
    if mode == 1:
        cv.putText(image, "MODE:" + mode_string[0], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def get_args():
    # 创建一个解析对象
    parser = argparse.ArgumentParser()
    # 下面为该对象添加参数
    parser.add_argument("--device", type=int, default=0)
    # 设置摄像机画面的高度和宽度
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    # 设置mediapipe模型需要的参数
    # 设置模式为静态图片读取
    parser.add_argument('--use_static_image_mode', action='store_true')
    # 设置最小的置信度
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)
    # 对该对象进行解析，并且返回结果
    args = parser.parse_args()
    return args


# 选择是否将数据写入到数据集之中
def select_mode(key, mode):
    number = -1
    # 0 ~ 9
    if 48 <= key <= 57:
        number = key - 48
    # 按c键切换模式,是否录入信息
    # mode 为0是不录入,mode 为1是录入
    if key == 99:
        mode = (mode + 1) % 2
    return number, mode


# 将数据保存到训练集中
def logging_csv(number, mode, landmark_list):
    if mode == 0:
        return
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/gesture_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


def run():
    # 得到参数解析的结果
    args = get_args()
    # 通过.运算符获取各个参数的值
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    # 设置摄像机的参数(宽,高)
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # 加载mediapipe的手部模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    gesture_classifier = GestureClassifier()
    # 读取csv文件,获取每个类别的名称
    with open('model/gesture_classifier/gesture_classifier_label.csv',
              encoding='utf-8-sig') as f:
        gesture_classifier_labels = csv.reader(f)
        gesture_classifier_labels = [
            row[0] for row in gesture_classifier_labels
        ]
    # 创建帧率对象的实例
    fps_cal = FpsCal(buffer_len=10)
    # 模式为0表示不录入数据,模式为1表示录入数据
    # 默认不录入数据
    mode = 0
    # 记录上一个手势的id
    pre_gesture = 0
    while True:
        fps = fps_cal.get()
        # 按下esc键结束
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        # 摄像机获取图片
        ret, image = cap.read()
        if not ret:
            break
        # 对图片进行镜像处理
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        # opencv为BGR通道，改为RGB通道
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        # 对图片进行训练，获取手部的特征点
        results = hands.process(image)
        image.flags.writeable = True
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 计算手部的矩形框位置

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 得到手部关键点的列表

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # 将这些关键点坐标进行标准化,使其可以输入到模型中进行预测

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # 将数据写入到数据集之中,该函数内部会通过mode判断是否写入

                logging_csv(number, mode, pre_processed_landmark_list)
                # 对处理后的关键点数据进行预测

                hand_sign_id = gesture_classifier(pre_processed_landmark_list)
                # 得到图像的大小
                image_width, image_height = debug_image.shape[1], debug_image.shape[0]

                relative_x = landmark_list[1][0] / image_width
                relative_y = landmark_list[1][1] / image_height
                # 获得屏幕的宽和高的像素数
                width, height = pyautogui.size()

                x = int((relative_x + 0.1) * int(width))
                y = int((relative_y + 0.2) * int(height))

                # 获取对于左、右手的检测结果
                is_left_hand = handedness.classification[0].label[0] == 'L'
                # 根据模型预测分类得到的结果进行相应的操作
                if is_left_hand:  # 判定为左手
                    # Press UP
                    if hand_sign_id == 5 and pre_gesture != 5:
                        pydirectinput.press('UP')
                    # Press DOWN
                    elif hand_sign_id == 6 and pre_gesture != 6:
                        pydirectinput.press('DOWN')
                    # Press LEFT
                    elif hand_sign_id == 7 and pre_gesture != 7:
                        pydirectinput.press('LEFT')
                    # Press RIGHT
                    elif hand_sign_id == 8 and pre_gesture != 8:
                        pydirectinput.press('RIGHT')
                # 判定为右手
                else:
                    # 鼠标移动
                    if hand_sign_id == 0:
                        pydirectinput.moveTo(x, y - 200, duration=0.1, _pause=False)
                        pass
                    # 点击左键
                    elif hand_sign_id == 3:
                        pydirectinput.click(button='left')
                        # 防止过快点击
                        time.sleep(1)
                    # 点击右键
                    elif hand_sign_id == 4:
                        pydirectinput.click(button='right')
                        # 防止过快点击
                        time.sleep(1)
                # 画出手部的矩形框
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = my_draw(debug_image, hand_landmarks)
                # 画文本框展示相关信息
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    gesture_classifier_labels[hand_sign_id],
                    "",
                )
                pre_gesture = hand_sign_id
        debug_image = draw_info(debug_image, fps, mode, number)
        # 展示图像
        cv.imshow('gesture recognition', debug_image)
    # 释放摄像机
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()
