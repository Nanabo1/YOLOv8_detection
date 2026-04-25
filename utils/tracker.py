# file: e:\PythonProject3\utils\tracker.py
"""
车辆跟踪模块 - 使用 DeepSORT算法进行多目标跟踪
"""
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(self, max_cosine_distance=0.4, nn_budget=None):
        """
        初始化 DeepSORT 跟踪器
        
        :param max_cosine_distance: 余弦距离阈值，用于外观特征匹配。值越小匹配越严格。
        :param nn_budget: 最近邻搜索预算，None表示使用所有历史特征。
        """
        # 【优化4】初始化 DeepSort 对象，重点开启 GPU 加速
        self.tracker = DeepSort(
            max_age=30,           # 最大丢失帧数，超过此值删除轨迹
            n_init=3,             # 连续检测到的帧数才确认为新轨迹
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None,
            embedder_gpu=True,    # 【关键】强制使用 GPU 进行特征提取 (RTX 3050 支持)
            embedder_model_name='mobilenet' # 使用轻量级模型 'mobilenet'
        )
        
        # 用于存储额外的元数据（如类别名称）
        self.track_metadata = {}

    def update(self, detections, frame=None):
        """
        更新跟踪状态
        :param detections: 当前帧检测到的车辆列表
        :param frame: 原始图像帧 (BGR格式)，用于内置Embedder提取特征。必须提供！
        :return: 更新后的跟踪结果列表
        """
        if not detections:
            # 如果没有检测结果，仍然调用 update 以处理轨迹丢失逻辑
            tracks = self.tracker.update_tracks([], frame=frame)
            self._clean_metadata()
            return []

        # 1. 预处理检测结果以适配 DeepSort
        deep_sort_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            if w < 0 or h < 0:
                continue
                
            deep_sort_detections.append((
                [x1, y1, w, h], 
                det['confidence'], 
                det['class_name']
            ))

        # 2. 执行跟踪更新
        if frame is None:
            raise ValueError("Frame must be provided when using internal embedder in VehicleTracker!")
            
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

        # 3. 构建返回结果
        results = []
        current_track_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            current_track_ids.add(track_id)
            
            # 获取边界框 [x1, y1, x2, y2]
            ltrb = track.to_ltrb() 
            x1, y1, x2, y2 = map(int, ltrb)
            
            # 查找当前 track 对应的最新检测信息以获取类别和置信度
            best_det = None
            max_iou = 0
            curr_bbox = [x1, y1, x2, y2]
            
            for det in detections:
                iou = self._calculate_iou(curr_bbox, det['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_det = det
            
            class_name = best_det['class_name'] if best_det else self.track_metadata.get(track_id, 'unknown')
            confidence = best_det['confidence'] if best_det else 0.0
            
            # 更新元数据缓存
            self.track_metadata[track_id] = class_name

            results.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'class_name': class_name,
                'confidence': confidence
            })

        # 清理不再存在的轨迹元数据
        self._clean_metadata(current_track_ids)

        return results

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的 IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area

    def _clean_metadata(self, active_ids=None):
        """清理无效轨迹的元数据"""
        if active_ids is None:
            self.track_metadata.clear()
        else:
            keys_to_remove = [k for k in self.track_metadata if k not in active_ids]
            for k in keys_to_remove:
                del self.track_metadata[k]

    def draw_tracks(self, frame, tracks, show_class=True):
        """
        在图像上绘制跟踪结果
        """
        if frame is None:
            return None

        result_frame = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            vehicle_id = track['id']
            class_name = track['class_name']

            # 根据 ID 生成固定颜色
            color = self._get_color_by_id(vehicle_id)
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # 构建标签文本
            if show_class:
                label = f'ID:{vehicle_id} {class_name}'
            else:
                label = f'ID:{vehicle_id}'
            
            # 绘制标签背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 确保标签不超出画面顶部
            text_y = y1 - 5
            bg_y1 = y1 - h - baseline - 5
            if bg_y1 < 0:
                bg_y1 = y1
                text_y = y1 + h + baseline
            
            cv2.rectangle(result_frame, (x1, bg_y1), (x1 + w, y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(result_frame, label, (x1, text_y - baseline),
                        font, font_scale, (255, 255, 255), thickness)

        return result_frame

    def _get_color_by_id(self, track_id):
        """
        根据 Track ID 生成一致的随机颜色
        """
        np.random.seed(int(track_id))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        np.random.seed(None)
        return color