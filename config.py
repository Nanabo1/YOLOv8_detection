# file: e:\PythonProject3\config.py
"""
配置文件 - 存放所有系统配置
"""


class Config:
    # ========== 模型配置 ==========
    # YOLOv8模型文件路径
    MODEL_PATH = 'models/best.pt'
    # 检测置信度阈值（0-1之间，越高越严格）
    CONFIDENCE_THRESHOLD = 0.5

    # ========== 数据库配置 ==========
    MYSQL_HOST = 'localhost'  # MySQL服务器地址
    MYSQL_PORT = 3306  # MySQL端口
    MYSQL_USER = 'root'  # MySQL用户名
    MYSQL_PASSWORD = '123456'  # 你的MySQL密码
    MYSQL_DB = 'vehicle_detection'  # 数据库名

    # 数据库连接URI
    # config.py 中的数据库配置
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
 
    # 摄像头帧率
    FPS = 30

    # ========== 透视变换配置 (用于提高测速精度) ==========
    # 源点：图像中梯形区域的四个角点 [左上, 右上, 右下, 左下]
    # 需要根据实际视频画面调整这些坐标！
    # 示例：假设画面中道路区域大致如下
    PERSPECTIVE_SRC_POINTS = [
        [220, 180],  # 左上
        [420, 180],  # 右上
        [620, 470],  # 右下
        [20, 470]    # 左下
    ]
    
    # 目标点：对应现实世界中的矩形区域 (单位：米)
    # 宽度设为10米，高度设为20米（代表视野深度）
    PERSPECTIVE_DST_POINTS = [
        [0, 0],      # 左上
        [4, 0],     # 右上
        [4, 25],    # 右下
        [0, 25]      # 左下
    ]

    # ========== 摄像头配置 ==========
    # 串口摄像头列表（USB摄像头）
    SERIAL_CAMERAS = [
        {'id': 0, 'name': '电脑自带摄像头'},
        {'id': 1, 'name': 'USB摄像头 1'},
        {'id': 2, 'name': 'USB摄像头 2'},
    ]

    # ========== 天气API配置 ==========
    # 和风天气API（免费版）
    WEATHER_API_KEY = 'q23qqrx9tu.re.qweatherapi.com'  # 注册后获取
    CITY_ID = '101180301'  # 城市ID
    WEATHER_API_URL = 'https://devapi.qweather.com/v7/weather/now'

    