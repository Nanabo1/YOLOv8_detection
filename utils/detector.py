# file: e:\PythonProject3\utils\detector.py
"""
车辆检测模块 - 使用YOLOv8识别车辆
"""
import cv2
import torch
from ultralytics import YOLO
import numpy as np


class VehicleDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        初始化检测器
        :param model_path: YOLO模型路径
        :param confidence_threshold: 置信度阈值
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        # 【优化1】优先使用 CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1:自行车, 2:汽车, 3:摩托车, 5:公交车, 7:卡车
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        # 加载模型
        try:
            print(f"正在加载模型: {model_path} on device: {self.device}")
            self.model = YOLO(model_path).to(self.device)
            
            # 【优化2】如果是CUDA设备，尝试转换为半精度(FP16)以加速推理并节省显存
            if self.device == 'cuda':
                self.model.half()
                print("模型已转换为 FP16 半精度模式")
                
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None

    def detect(self, frame):
        """
        检测图像中的车辆
        :param frame: 输入图像
        :return: 检测结果列表
        """
        if self.model is None:
            print("模型未加载，无法检测")
            return []

        if frame is None:
            return []

        try:
            # 【优化3】关键参数调整
            # imgsz=640: 固定输入尺寸，避免处理4K等大图导致显存爆炸
            # half=True: 再次确保使用半精度 (如果init中失败，这里作为备份)
            # verbose=False: 关闭日志输出，减少IO开销
            results = self.model(
                frame, 
                verbose=False, 
                device=self.device, 
                imgsz=640, 
                half=(self.device == 'cuda')
            )[0]

            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # 获取置信度
                    confidence = float(box.conf[0].cpu().numpy())
                    # 获取类别
                    class_id = int(box.cls[0].cpu().numpy())

                    # 只保留车辆类别且置信度大于阈值
                    if class_id in self.vehicle_classes and confidence > self.confidence_threshold:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, 'unknown')
                        })

            return detections

        except Exception as e:
            print(f"检测过程出错: {e}")
            return []

    def draw_detections(self, frame, detections):
        """
        在图像上绘制检测结果
        :param frame: 原始图像
        :param detections: 检测结果
        :return: 绘制后的图像
        """
        if frame is None:
            return None

        result_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # 绘制边界框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(result_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_frame