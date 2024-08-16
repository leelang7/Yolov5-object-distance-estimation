import torch
import numpy as np
import yolov5
import cv2
import math
from time import time
from collections import deque

class DistanceEstimationDetector:
    def __init__(self, video_path, model_path):
        """
        :param video_path: 처리할 영상
        :param model_path: 모델
        """
        self.video_path = video_path
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.frame_times = deque(maxlen=30)  # Store times of last 30 frames

    def get_video_capture(self):
        """
        :return: 비디오 리턴
        """
        return cv2.VideoCapture(self.video_path)

    def load_model(self, model_path):
        """
        Model 로드 및 구성
        :param model_path: 로드할 모델 파일 경로
        :return: 로드 및 구성된 모델
        """
        model = yolov5.load(model_path)
        model.conf = 0.40  # confidence 임계값
        model.iou = 0.45  # NMS IoU threshold
        model.max_det = 1000  # 최대 감지 수(한 프레임당)
        model.classes = [0, 2]  # 객체의 class number: 사람과 자동차만 검출
        return model

    def get_model_results(self, frame):
        """
        예측 및 예측 결과 반환
        :param frame: 동영상 프레임
        :return: 입력 프레임의 결과
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame, size=640)
        predictions = results.xyxyn[0]
        cords, scores, labels = predictions[:, :4], predictions[:, 4], predictions[:, 5]
        return cords, scores, labels

    def draw_rect(self, results, frame):
        """
        바운딩 박스 그리기
        :param results: 모델에서 반환된 객체 감지 결과
        :param frame: 처리중인 프레임
        :return: 바운딩 박스처리된 이미지
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        n = len(labels)  # 감지된 개체(인스턴스) 수

        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            green_bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), green_bgr, 1)  # 바운딩 박스 그리기
            cls = labels[i]  # 클래스 네임 얻기
            cls = int(cls)
            cls_name = ""
            if cls == 2:
                cls_name = 'car'
            elif cls == 0:
                cls_name = 'person'
            cv2.putText(frame, cls_name, (x1 + 35, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)

        return frame

    def calc_distances(self, results, frame):
        """
        거리 계산
        :param results: 객체 감지로 얻은 결과 값
        :param frame: 처리할 프레임
        :return: 거리 계산을 수행하고 처리한 이미지
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        points = []

        # 거리 계산용 변수 (임의의 예제 값 사용)
        real_height = 60  # 실제 물체 높이 (inch)
        focal_length = 700  # 초점 거리 (임의 단위)

        for car in cord:
            x1, y1, x2, y2 = int(car[0] * x_shape), int(car[1] * y_shape), int(car[2] * x_shape), int(car[3] * y_shape)
            x_mid_rect, y_mid_rect = (x1 + x2) / 2, (y1 + y2) / 2
            y_line_length, x_line_length = abs(y1 - y2), abs(x1 - x2)
            points.append([x1, y1, x2, y2, int(x_mid_rect), int(y_mid_rect), int(x_line_length), int(y_line_length)])

        for i in range(0, len(points)):
            end_x1, end_y1, end_x2, end_y2, end_x_mid_rect, end_y_mid_rect, end_x_line_length, end_y_line_length = points[i]

            # 거리 계산 예제 (고급 알고리즘 적용 가능)
            if end_y_line_length != 0:  # 0으로 나누지 않도록 보호
                distance = real_height * focal_length / end_y_line_length
            else:
                distance = 0

            cv2.putText(frame, f"{round(distance, 2)} m", (int(end_x1), int(end_y2)), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(frame, str(int(scores[i] * 100)) + "%", (int(end_x1), int(end_y1)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 2)
        return frame

    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()
        while True:
            ret, frame = cap.read()
            assert ret
            start_time = time()

            results = self.get_model_results(frame)
            frame = self.draw_rect(results, frame)
            frame = self.calc_distances(results, frame)

            end_time = time()
            inference_time = end_time - start_time
            self.frame_times.append(inference_time)
            fps = 1 / np.mean(self.frame_times)

            # 탐지된 객체 정보 출력
            cords, scores, labels = results
            detection_info = []  # Detection information list
            for i in range(len(labels)):
                cls_id = int(labels[i])
                cls_name = self.classes[cls_id]
                score = scores[i]
                detection_info.append(f"{cls_name} ({score:.2f})")
            
            # 표준 출력
            if detection_info:
                print(f"FPS: {fps:.2f} | Inference Time: {inference_time:.4f}s | Detected Objects: {', '.join(detection_info)}")
            else:
                print(f"FPS: {fps:.2f} | Inference Time: {inference_time:.4f}s | No Objects Detected")

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv5 Distance Estimation', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# DistanceEstimationDetector 객체 생성 및 시작
detector = DistanceEstimationDetector(video_path='input/car_input1.mp4', model_path='yolov5s.pt')
detector()
