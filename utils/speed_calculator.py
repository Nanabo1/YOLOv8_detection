# file: e:\PythonProject3\utils\speed_calculator.py
"""
速度计算模块 - 优化版：针对低配设备，降低计算频率以平滑速度
"""
import numpy as np
import time


class SpeedCalculator:
    def __init__(self, perspective_transformer, fps=30):
        """
        初始化速度计算器
        :param perspective_transformer: PerspectiveTransformer 实例
        :param fps: 视频帧率 (用于辅助判断，主要依赖时间戳)
        """
        self.transformer = perspective_transformer
        self.fps = fps
        
        # 存储每个车辆上次用于计算速度的状态
        # 结构: {vehicle_id: {'pos': (x, y), 'timestamp': float}}
        self.last_calc_points = {} 
        
        self.stats = {}  # 统计信息
        
        # 【优化5】关键优化：增大最小时间间隔
        # RTX 3050 在运行 YOLO+DeepSORT 时，实际帧率可能波动在 10-20 FPS。
        # 如果间隔太短（如0.1s），像素级的检测抖动会被放大成巨大的速度误差。
        # 设置为 1.0 秒意味着我们计算的是“过去1秒的平均速度”，这更稳定。
        self.min_time_interval = 1.0 

    def calculate_speed(self, vehicle_id, bbox, timestamp=None):
        """
        计算车辆速度
        :param vehicle_id: 车辆ID
        :param bbox: 边界框 [x1, y1, x2, y2]
        :param timestamp: 时间戳
        :return: 当前速度 km/h，如果时间间隔不足则返回 0.0
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 获取现实世界坐标 (米)
        real_pos = self.transformer.transform_bbox_center(bbox)
        
        if real_pos is None:
            return 0.0
            
        current_pos = real_pos # (meter_x, meter_y)

        # 2. 检查是否有历史记录
        if vehicle_id not in self.last_calc_points:
            # 第一次见到该车，记录初始点，不计算速度
            self.last_calc_points[vehicle_id] = {
                'pos': current_pos,
                'timestamp': timestamp
            }
            return 0.0

        last_data = self.last_calc_points[vehicle_id]
        last_pos = last_data['pos']
        last_time = last_data['timestamp']

        # 3. 计算时间差
        time_diff = timestamp - last_time

        # 【优化6】如果时间间隔小于设定值，跳过计算
        if time_diff < self.min_time_interval:
            return 0.0

        # 4. 计算位移 (米)
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        distance_meters = np.sqrt(dx ** 2 + dy ** 2)

        # 【优化7】最小位移过滤：排除静止时的像素抖动
        # 如果1秒钟移动不到0.5米，视为静止，不更新速度，避免显示 2-5 km/h 的幽灵速度
        if distance_meters < 0.5:
             self.last_calc_points[vehicle_id] = {
                'pos': current_pos,
                'timestamp': timestamp
            }
             return 0.0

        # 5. 计算速度
        speed_kmh = 0.0
        if time_diff > 0:
            speed_mps = distance_meters / time_diff
            speed_kmh = speed_mps * 3.6

        # 6. 合理性过滤 (例如：排除超光速异常)
        if 2 < speed_kmh < 200:
            # 更新统计信息
            if vehicle_id not in self.stats:
                self.stats[vehicle_id] = {
                    'max': speed_kmh,
                    'min': speed_kmh,
                    'avg': speed_kmh,
                    'current': speed_kmh,
                    'count': 1
                }
            else:
                s = self.stats[vehicle_id]
                s['max'] = max(s['max'], speed_kmh)
                s['min'] = min(s['min'], speed_kmh)
                # 简单移动平均，让平均值更平滑
                s['avg'] = (s['avg'] * s['count'] + speed_kmh) / (s['count'] + 1)
                s['current'] = speed_kmh
                s['count'] += 1
            
            # 【关键】更新参考点为当前点，作为下一次计算的起点
            self.last_calc_points[vehicle_id] = {
                'pos': current_pos,
                'timestamp': timestamp
            }
            
            return speed_kmh
        
        # 如果速度异常，也更新参考点，防止后续计算基于过旧的数据
        self.last_calc_points[vehicle_id] = {
            'pos': current_pos,
            'timestamp': timestamp
        }
        return 0.0

    def get_stats(self):
        """
        获取所有车辆的统计信息
        :return: 统计信息字典
        """
        return self.stats

    def clear_old_data(self, max_age_seconds=10):
        """
        清理长时间未出现的车辆数据，防止内存泄漏
        """
        current_time = time.time()
        ids_to_remove = []
        for vid, data in self.last_calc_points.items():
            if current_time - data['timestamp'] > max_age_seconds:
                ids_to_remove.append(vid)
        
        for vid in ids_to_remove:
            if vid in self.last_calc_points:
                del self.last_calc_points[vid]
            if vid in self.stats:
                del self.stats[vid]