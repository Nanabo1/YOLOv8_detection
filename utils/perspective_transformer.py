# file: e:\PythonProject3\utils\perspective_transformer.py
"""
透视变换模块 -将图像像素坐标转换为现实世界坐标(米)
"""
import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(self, src_points, dst_points):
        """
        初始化透视变换器
        :param src_points: 图像中的源点列表 [[x1,y1], [x2,y2], ...]
        :param dst_points: 现实世界中的目标点列表 [[x1,y1], [x2,y2], ...] (单位:米)
        """
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        
        # 计算透视变换矩阵
        self.update_matrix()

    def update_matrix(self):
        """重新计算透视变换矩阵"""
        try:
            self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.inv_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
            return True
        except Exception as e:
            print(f"更新透视矩阵失败: {e}")
            return False

    def update_config(self, src_points, dst_points):
        """外部调用此方法更新配置"""
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        return self.update_matrix()

    def transform_point(self, point):
        """
        将单个像素点 (x, y) 转换为现实世界坐标 (meter_x, meter_y)
        :param point: tuple (x, y)
        :return: tuple (meter_x, meter_y)
        """
        if point is None:
            return None
            
        # 准备输入数据格式
        pt = np.float32([[point]])
        
        # 执行透视变换
        transformed_pt = cv2.perspectiveTransform(pt, self.matrix)
        
        # 提取结果
        meter_x, meter_y = transformed_pt[0][0]
        return (meter_x, meter_y)

    def transform_bbox_center(self, bbox):
        """
        计算边界框中心点并转换为现实世界坐标
        :param bbox: [x1, y1, x2, y2]
        :return: (meter_x, meter_y) 或 None
        """
        if not bbox or len(bbox) != 4:
            return None
            
        x1, y1, x2, y2 = bbox
        
        # 注意：对于车辆测速，通常使用车辆底部中心点更准确，因为那是接触地面的点
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2 
        
        return self.transform_point((bottom_center_x, bottom_center_y))

    def draw_roi(self, frame):
        """
        在帧上绘制感兴趣区域(ROI)，用于调试校准
        :param frame: 原始图像
        :return: 绘制后的图像
        """
        if frame is None:
            return None
            
        result_frame = frame.copy()
        pts = self.src_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(result_frame, [pts], True, (0, 255, 255), 2)
        
        # 标注点
        for i, pt in enumerate(self.src_points):
            cv2.circle(result_frame, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(result_frame, str(i), tuple(pt.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
        return result_frame