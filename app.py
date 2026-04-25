"""
主程序 - Flask应用入口
"""
from flask import Flask, render_template, request, jsonify, Response, session, make_response, redirect, url_for, flash, \
    has_request_context
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import json
import cv2
import threading
import requests
import time
import base64
from config import Config
from utils.detector import VehicleDetector
from utils.tracker import VehicleTracker
from utils.speed_calculator import SpeedCalculator
from utils.perspective_transformer import PerspectiveTransformer  # 新增导入
import logging
import os
import csv
import io
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# 初始化Flask应用
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'vehicle_detection_secret_key_2024'
db = SQLAlchemy(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 免费天气API配置（使用Open-Meteo）==========
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# ========== 用户角色常量 ==========
ROLE_USER = 'user'
ROLE_ADMIN = 'admin'


# ========== 数据库模型 ==========
class User(db.Model):
    """用户表"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_USER)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'last_login': self.last_login.strftime('%Y-%m-%d %H:%M:%S') if self.last_login else None
        }


class VehicleRecord(db.Model):
    """车辆记录表"""
    __tablename__ = 'vehicle_records'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    vehicle_id = db.Column(db.String(50), nullable=False)
    speed = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    location = db.Column(db.String(100), nullable=True)
    vehicle_type = db.Column(db.String(50), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'speed': self.speed,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'location': self.location,
            'vehicle_type': self.vehicle_type
        }


class SpeedStats(db.Model):
    """速度统计表"""
    __tablename__ = 'speed_stats'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    vehicle_id = db.Column(db.String(50), nullable=False)
    max_speed = db.Column(db.Float, nullable=True)
    min_speed = db.Column(db.Float, nullable=True)
    avg_speed = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'max_speed': self.max_speed,
            'min_speed': self.min_speed,
            'avg_speed': self.avg_speed,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }


class DetectionLog(db.Model):
    """检测日志表 - 增加用户信息"""
    __tablename__ = 'detection_logs'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    log_level = db.Column(db.String(20), default='INFO')
    message = db.Column(db.Text, nullable=False)
    username = db.Column(db.String(50), nullable=True)
    user_role = db.Column(db.String(20), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'log_level': self.log_level,
            'message': self.message,
            'username': self.username,
            'user_role': self.user_role,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

# ========== 登录装饰器 ==========
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login'))

        user = db.session.get(User, session['user_id'])
        if not user or user.role != ROLE_ADMIN:
            flash('需要管理员权限', 'danger')
            return redirect(url_for('index'))

        return f(*args, **kwargs)

    return decorated_function


# ========== 初始化组件 ==========
try:
    detector = VehicleDetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)
    tracker = VehicleTracker()
    
    # 初始化透视变换器
    perspective_transformer = PerspectiveTransformer(
        src_points=Config.PERSPECTIVE_SRC_POINTS,
        dst_points=Config.PERSPECTIVE_DST_POINTS
    )
    
    # 初始化速度计算器，传入透视变换器
    speed_calculator = SpeedCalculator(perspective_transformer, Config.FPS)
    
    print("所有组件初始化成功")
except Exception as e:
    print(f"组件初始化失败: {e}")
    detector = None
    tracker = None
    speed_calculator = None
    perspective_transformer = None

# ========== 全局变量 ==========
camera = None
is_detecting = False
current_frame = None
frame_lock = threading.Lock()
action_logs = []  # 内存中的动作日志
detection_results = {
    'tracks': [],
    'count': 0,
    'stats': {}
}
speed_stats_dict = {}
camera_type = 'local'
camera_source = 0
video_file_path = None
current_mode = 'detect'  # 'detect' 或 'speed'
video_thread_running = False
frame_counter = 0
last_save_time = time.time()
last_display_frame = None
last_detection_result = None
last_detection_time = 0
DETECTION_INTERVAL = 1.0# 每 0.1 秒检测一帧
MAX_STREAM_FPS = 15

# ========== 帧跳过策略配置 ==========
# 每隔 N 帧进行一次检测。
# 例如：值为 2 表示每 3 帧检测 1 帧 (0, 3, 6...)，跳过 2 帧。
# 在 30FPS 视频流下，设置为 2 意味着检测频率约为 10FPS，能大幅降低延迟。
FRAME_SKIP_INTERVAL = 2 


# ========== 辅助函数 ==========
def get_current_user():
    """获取当前用户信息（安全处理请求上下文）"""
    if not has_request_context():
        return None, None

    try:
        if 'user_id' in session:
            with app.app_context():
                user = db.session.get(User, session['user_id'])
                if user:
                    return user.username, user.role
    except Exception as e:
        print(f"获取用户信息失败: {e}")

    return None, None


def save_log_to_db(message, level='INFO', username=None, user_role=None):
    """保存日志到数据库（带应用上下文）"""
    try:
        with app.app_context():
            log_record = DetectionLog(
                log_level=level,
                message=message,
                username=username,
                user_role=user_role
            )
            db.session.add(log_record)
            db.session.commit()
            return True
    except Exception as e:
        print(f"保存日志到数据库失败: {e}")
        return False


def save_vehicle_record_to_db(vehicle_id, speed, vehicle_type):
    """保存车辆记录到数据库（带应用上下文）"""
    try:
        with app.app_context():
            record = VehicleRecord(
                vehicle_id=f'vehicle_{vehicle_id}',
                speed=speed,
                vehicle_type=vehicle_type,
                location='默认位置'
            )
            db.session.add(record)
            db.session.commit()
            print(f"车辆记录已保存: ID={vehicle_id}, 速度={speed}, 类型={vehicle_type}")
            return True
    except Exception as e:
        print(f"保存车辆记录失败: {e}")
        return False


def save_speed_stats_to_db(vehicle_id, max_speed, min_speed, avg_speed):
    """保存速度统计到数据库（带应用上下文）"""
    try:
        with app.app_context():
            stat = SpeedStats(
                vehicle_id=f'vehicle_{vehicle_id}',
                max_speed=max_speed,
                min_speed=min_speed,
                avg_speed=avg_speed
            )
            db.session.add(stat)
            db.session.commit()
            print(f"速度统计已保存: ID={vehicle_id}, 最高={max_speed}, 最低={min_speed}, 平均={avg_speed}")
            return True
    except Exception as e:
        print(f"保存速度统计失败: {e}")
        return False


def add_log(message, level='INFO'):
    """添加日志到内存"""
    global action_logs

    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f'[{timestamp}] {message}'
    action_logs.append(log_entry)
    if len(action_logs) > 100:
        action_logs = action_logs[-100:]

    # 获取当前用户信息
    username, user_role = get_current_user()

    # 异步保存到数据库（传入用户信息）
    def async_save():
        save_log_to_db(message, level, username, user_role)

    threading.Thread(target=async_save).start()

    logger.info(message)
    return log_entry


def get_weather():
    """获取天气信息（使用免费Open-Meteo API）"""
    try:
        default_city = "新乡"

        # 通过城市名称获取经纬度
        geocode_params = {
            "name": default_city,
            "count": 1,
            "language": "zh",
            "format": "json"
        }

        geocode_response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params=geocode_params,
            timeout=5
        )

        if geocode_response.status_code == 200:
            geocode_data = geocode_response.json()
            if geocode_data.get("results") and len(geocode_data["results"]) > 0:
                location = geocode_data["results"][0]
                lat = location["latitude"]
                lon = location["longitude"]
                city_name = location.get("name", default_city)

                # 获取天气数据
                weather_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m",
                    "timezone": "Asia/Shanghai",
                    "forecast_days": 1
                }

                weather_response = requests.get(
                    OPEN_METEO_WEATHER_URL,
                    params=weather_params,
                    timeout=5
                )

                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    current = weather_data.get("current_weather", {})
                    hourly = weather_data.get("hourly", {})

                    # 获取当前小时的湿度
                    current_time = datetime.now().strftime("%Y-%m-%dT%H:00")
                    humidity = 45
                    if hourly.get("time") and hourly.get("relativehumidity_2m"):
                        for i, t in enumerate(hourly["time"]):
                            if t.startswith(current_time[:13]):
                                humidity = hourly["relativehumidity_2m"][i]
                                break

                    wind_dir = get_wind_direction(current.get("winddirection", 0))

                    return {
                        'temp': str(int(current.get("temperature", 22))),
                        'text': get_weather_description(current.get("weathercode", 0)),
                        'windDir': wind_dir,
                        'windScale': str(int(current.get("windspeed", 10) / 3.6)),
                        'humidity': str(int(humidity)),
                        'city': city_name
                    }

        return {
            'temp': '22',
            'text': '晴',
            'windDir': '东南风',
            'windScale': '3',
            'humidity': '45',
            'city': '北京'
        }

    except Exception as e:
        print(f'获取天气失败: {e}')
        return {
            'temp': '22',
            'text': '晴',
            'windDir': '东南风',
            'windScale': '3',
            'humidity': '45',
            'city': '北京'
        }


def get_weather_description(weather_code):
    """将Open-Meteo天气代码转换为文字描述"""
    weather_map = {
        0: "晴天",
        1: "多云",
        2: "多云",
        3: "阴天",
        45: "雾",
        48: "雾",
        51: "小雨",
        53: "小雨",
        55: "中雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        95: "雷雨"
    }
    return weather_map.get(weather_code, "多云")


def get_wind_direction(degrees):
    """将风向角度转换为文字方向"""
    directions = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]
    index = int((degrees + 22.5) / 45) % 8
    return directions[index]

# app.py -> process_frame 函数

def process_frame(frame):
    """处理单帧图像 - 优化版：增加类别显示"""
    global detection_results, speed_stats_dict, current_mode, detector, tracker, speed_calculator
    global frame_counter, perspective_transformer
    
    if frame is None:
        return None

    # 统一缩放到 640x480 进行处理和显示
    display_frame = cv2.resize(frame, (640, 480))

    if is_detecting and detector is not None:
        try:
            # 1. 执行检测
            detections = detector.detect(display_frame)
            # 2. 执行跟踪
            tracks = tracker.update(detections, frame=display_frame)
            
            # 3. 创建副本用于绘制
            final_frame = display_frame.copy()
            
            # 遍历跟踪结果进行绘制
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                vehicle_id = track['id']
                class_name = track.get('class_name', 'unknown') # 安全获取类别
                
                # 【优化1】过滤掉无效的跟踪框（例如类别未知或置信度极低）
                if not class_name or class_name == 'unknown':
                    continue

                # --- 绘制基础边界框和ID ---
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                
                # 组合文本: ID 和 类别
                label_text = f"ID:{vehicle_id} {class_name}"
                
                # 计算文本大小以绘制背景
                (w, h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # 绘制黑色半透明背景
                bg_y1 = max(y1 - h - 5, 0)
                bg_y2 = y1 + baseline + 2
                cv2.rectangle(final_frame, (x1, bg_y1), (x1 + w + 5, bg_y2), (0, 0, 0), -1)
                
                # 绘制白色文字
                text_y = y1 - 2
                if text_y < h: 
                    text_y = y1 + h + baseline
                cv2.putText(final_frame, label_text, (x1 + 2, text_y),
                            font, font_scale, (255, 255, 255), thickness)

            # 测速模式特有逻辑
            if current_mode == 'speed' and speed_calculator is not None:
                # 绘制透视校准区域 (黄色)
                if perspective_transformer:
                    final_frame = perspective_transformer.draw_roi(final_frame)
                
                current_time = time.time()
                speeds = {}
                
                # 【优化2】再次遍历用于测速显示（或者合并到上面循环，但分开更清晰）
                # 注意：这里我们重新遍历 tracks，因为需要计算速度
                for track in tracks:
                    vehicle_id = track['id']
                    bbox = track['bbox']
                    class_name = track.get('class_name', 'unknown')
                    
                    if class_name == 'unknown':
                        continue

                    # 计算速度
                    speed = speed_calculator.calculate_speed(vehicle_id, bbox, current_time)
                    
                    # 只有当速度有效时才绘制速度标签，避免满屏都是 "0.0 km/h" 或空白
                    if speed > 0:
                        speeds[vehicle_id] = speed
                        
                        # 定义颜色变量
                        if speed < 60:
                            color = (0, 255, 0)      # 绿色
                        elif speed < 100:
                            color = (0, 255, 255)    # 黄色
                        else:
                            color = (0, 0, 255)      # 红色

                        speed_text = f'{speed:.1f} km/h'
                        
                        x1, y1, x2, y2 = bbox
                        (w_sp, h_sp), bl_sp = cv2.getTextSize(speed_text, font, font_scale, thickness)
                        text_x = x1
                        text_y = y2 + h_sp + 2
                        
                        # 防止超出底部
                        if text_y > 480:
                             text_y = y2 - 2
                             
                        # 速度背景
                        cv2.rectangle(final_frame, (text_x, text_y - h_sp - 2), (text_x + w_sp, text_y + bl_sp), (0, 0, 0), -1)
                        # 绘制速度文字
                        cv2.putText(final_frame, speed_text, (text_x, text_y),
                                    font, font_scale, color, thickness)
                        
                        # 异步保存数据
                        if frame_counter % 30 == 0:
                            threading.Thread(target=save_vehicle_record_to_db, args=(vehicle_id, speed, class_name)).start()

                speed_stats_dict = speed_calculator.get_stats()
                
                # 定期清理旧数据
                if frame_counter % 100 == 0:
                    speed_calculator.clear_old_data()
                
                # 定期保存统计
                if frame_counter % 60 == 0:
                    for vid, stats in speed_stats_dict.items():
                        threading.Thread(target=save_speed_stats_to_db, args=(vid, stats['max'], stats['min'], stats['avg'])).start()

                detection_results = {
                    'tracks': tracks,
                    'speeds': speeds,
                    'stats': speed_stats_dict,
                    'count': len(tracks)
                }
            else:
                # 识别模式数据保存
                if frame_counter % 30 == 0:
                    for track in tracks:
                        if track.get('class_name', 'unknown') != 'unknown':
                            threading.Thread(target=save_vehicle_record_to_db, args=(track['id'], 0, track['class_name'])).start()
                
                detection_results = {
                    'tracks': tracks,
                    'count': len(tracks),
                    'stats': {}
                }
            
            frame_counter += 1
            return final_frame

        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            detection_results = {'tracks': [], 'count': 0, 'stats': {}}
    else:
        detection_results = {'tracks': [], 'count': 0, 'stats': {}}
        
    return display_frame

# app.py -> video_stream 函数优化

def video_stream():
    """视频流生成器 - 优化版：降低编码质量和检测频率"""
    global camera, is_detecting, video_thread_running, frame_counter, last_display_frame, last_detection_time
    
    print("视频流生成器启动")
    video_thread_running = True
    
    stream_frame_count = 0
    last_frame_time = time.time()

    while video_thread_running:
        try:
            current_time = time.time()
            
            # 1. 控制推流帧率 (避免 CPU 100% 编码)
            if current_time - last_frame_time < 1.0 / MAX_STREAM_FPS:
                time.sleep(0.01)
                continue
            
            if camera is not None and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    processed_frame = None
                    
                    # 2. 检测逻辑 (独立于推流频率)
                    if is_detecting and detector is not None:
                        # 每隔 DETECTION_INTERVAL 秒执行一次完整检测
                        if current_time - last_detection_time >= DETECTION_INTERVAL:
                            last_detection_time = current_time
                            # 执行检测和跟踪
                            processed_frame = process_frame(frame)
                            if processed_frame is not None:
                                last_display_frame = processed_frame
                        else:
                            # 在非检测帧，直接使用上一帧的检测结果画面
                            # 这样既保持了画面流畅，又避免了重复推理
                            if last_display_frame is not None:
                                processed_frame = last_display_frame
                            else:
                                # 如果还没有检测结果，至少显示原图
                                processed_frame = cv2.resize(frame, (640, 480))
                    else:
                        # 未开启检测，直接缩放显示
                        processed_frame = cv2.resize(frame, (640, 480))
                        last_display_frame = processed_frame # 更新缓存

                    # 3. 编码并发送
                    if processed_frame is not None:
                        # 【关键优化】降低 JPEG 质量到 60，大幅减少 CPU 编码时间
                        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            
                            last_frame_time = time.time()
                            stream_frame_count += 1
                else:
                    # 视频结束处理
                    if video_file_path:
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(0.1)
            else:
                time.sleep(0.5)
        except Exception as e:
            print(f"视频流错误: {e}")
            time.sleep(0.5)

    print("视频流生成器结束")
# ========== 用户认证路由 ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_role'] = user.role
            user.last_login = datetime.now()
            db.session.commit()

            add_log(f'用户登录: {username} (角色: {user.role})', 'INFO')
            flash('登录成功', 'success')

            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not password:
            flash('用户名和密码不能为空', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('两次输入的密码不一致', 'danger')
            return render_template('register.html')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('用户名已存在', 'danger')
            return render_template('register.html')

        new_user = User(username=username, role=ROLE_USER)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        add_log(f'新用户注册: {username}', 'INFO')
        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    """用户登出"""
    username = session.get('username')
    session.clear()
    if username:
        add_log(f'用户登出: {username}', 'INFO')
    flash('已退出登录', 'info')
    return redirect(url_for('login'))


# ========== 管理员路由 ==========
@app.route('/admin/users')
@admin_required
def admin_users():
    """管理员用户管理页面"""
    users = User.query.all()
    return render_template('admin_users.html', users=users)


@app.route('/api/admin/users', methods=['GET'])
@admin_required
def api_get_users():
    """获取所有用户"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])


@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def api_update_user(user_id):
    """更新用户信息"""
    user = User.query.get_or_404(user_id)
    data = request.json

    if 'username' in data and data['username'] != user.username:
        existing = User.query.filter_by(username=data['username']).first()
        if existing:
            return jsonify({'success': False, 'message': '用户名已存在'})
        user.username = data['username']

    if 'password' in data and data['password']:
        user.set_password(data['password'])

    if 'role' in data:
        old_role = user.role
        user.role = data['role']
        add_log(f'管理员修改用户角色: {user.username} 从 {old_role} 变更为 {data["role"]}', 'INFO')

    if 'reset_password' in data and data['reset_password']:
        user.set_password('123456')
        add_log(f'管理员重置用户密码: {user.username}', 'WARNING')

    db.session.commit()

    return jsonify({'success': True, 'message': '用户信息更新成功'})


@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def api_delete_user(user_id):
    """删除用户"""
    if user_id == session['user_id']:
        return jsonify({'success': False, 'message': '不能删除当前登录的管理员账户'})

    user = User.query.get_or_404(user_id)
    username = user.username
    db.session.delete(user)
    db.session.commit()

    add_log(f'管理员删除用户: {username}', 'WARNING')
    return jsonify({'success': True, 'message': '用户已删除'})


@app.route('/api/admin/users', methods=['POST'])
@admin_required
def api_create_user():
    """创建新用户（管理员创建）"""
    data = request.json
    username = data.get('username')
    password = data.get('password', '123456')
    role = data.get('role', ROLE_USER)

    if not username:
        return jsonify({'success': False, 'message': '用户名不能为空'})

    existing = User.query.filter_by(username=username).first()
    if existing:
        return jsonify({'success': False, 'message': '用户名已存在'})

    new_user = User(username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    add_log(f'管理员创建用户: {username} (角色: {role})', 'INFO')
    return jsonify({'success': True, 'message': '用户创建成功', 'user': new_user.to_dict()})


@app.route('/api/admin/users/<int:user_id>/promote', methods=['POST'])
@admin_required
def api_promote_user(user_id):
    """提升用户为管理员"""
    user = User.query.get_or_404(user_id)
    user.role = ROLE_ADMIN
    db.session.commit()

    add_log(f'管理员提升用户为管理员: {user.username}', 'INFO')
    return jsonify({'success': True, 'message': f'用户 {user.username} 已提升为管理员'})


# ========== 路由 ==========
@app.route('/')
@login_required
def index():
    """首页"""
    weather = get_weather()
    return render_template('index.html', weather=weather)


@app.route('/detect')
@login_required
def detect():
    """车辆识别页面"""
    global current_mode
    current_mode = 'detect'
    add_log('进入车辆识别页面')
    return render_template('detect.html')


@app.route('/detect_speed')
@login_required
def detect_speed():
    """速度检测页面"""
    global current_mode
    current_mode = 'speed'
    add_log('进入速度检测页面')
    return render_template('detect_speed.html')


@app.route('/history')
@login_required
def history():
    """历史记录页面"""
    add_log('访问历史记录页面')
    return render_template('history.html')


@app.route('/charts')
@login_required
def charts():
    """数据统计页面"""
    add_log('访问数据统计页面')
    return render_template('charts.html')


@app.route('/settings')
@login_required
def settings():
    """系统设置页面"""
    add_log('访问系统设置页面')
    return render_template('settings.html')


@app.route('/about')
def about():
    """关于我们页面"""
    return render_template('about.html')


@app.route('/api/logs')
@login_required
def get_logs():
    """获取内存中的日志"""
    return jsonify({'logs': action_logs})


@app.route('/api/database_logs')
@login_required
def get_database_logs():
    """获取数据库中的日志"""
    try:
        with app.app_context():
            logs = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).limit(100).all()
            return jsonify([log.to_dict() for log in logs])
    except Exception as e:
        print(f"获取数据库日志失败: {e}")
        return jsonify([])


@app.route('/api/detection_results')
@login_required
def get_detection_results():
    """获取检测结果"""
    return jsonify(detection_results)


@app.route('/api/vehicle_records')
@login_required
def get_vehicle_records():
    """获取车辆记录"""
    try:
        with app.app_context():
            records = VehicleRecord.query.order_by(VehicleRecord.timestamp.desc()).limit(100).all()
            return jsonify([r.to_dict() for r in records])
    except Exception as e:
        print(f"获取车辆记录失败: {e}")
        return jsonify([])


@app.route('/api/speed_stats')
@login_required
def get_speed_stats():
    """获取速度统计"""
    try:
        with app.app_context():
            stats = SpeedStats.query.order_by(SpeedStats.timestamp.desc()).limit(50).all()
            return jsonify([s.to_dict() for s in stats])
    except Exception as e:
        print(f"获取速度统计失败: {e}")
        return jsonify([])


@app.route('/api/start_camera', methods=['POST'])
@login_required
def start_camera():
    """启动摄像头"""
    global camera, camera_type, camera_source, is_detecting, video_file_path, video_thread_running

    data = request.json
    cam_type = data.get('type', 'local')
    source = data.get('source', 0)

    try:
        if camera is not None:
            camera.release()
            camera = None
            time.sleep(0.5)

        print(f"尝试打开摄像头: {cam_type} - {source}")
        if cam_type == 'local':
            camera = cv2.VideoCapture(int(source))
        elif cam_type == 'tcp':
            camera = cv2.VideoCapture(source)
        elif cam_type == 'serial':
            camera = cv2.VideoCapture(int(source))

        if camera is not None and camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            camera_type = cam_type
            camera_source = source
            video_file_path = None
            is_detecting = False
            video_thread_running = True

            add_log(f'摄像头启动成功: {cam_type} - {source}')
            return jsonify({'success': True, 'message': '摄像头启动成功'})
        else:
            return jsonify({'success': False, 'message': '摄像头启动失败'})

    except Exception as e:
        print(f"启动摄像头异常: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/stop_camera', methods=['POST'])
@login_required
def stop_camera():
    """停止摄像头"""
    global camera, is_detecting, video_file_path, video_thread_running

    video_thread_running = False
    time.sleep(0.5)

    if camera is not None:
        camera.release()
        camera = None

    is_detecting = False
    video_file_path = None

    add_log('摄像头已关闭')
    return jsonify({'success': True})


@app.route('/api/start_detection', methods=['POST'])
@login_required
def start_detection():
    """开始检测"""
    global is_detecting, frame_counter

    if camera is None:
        return jsonify({'success': False, 'message': '请先启动摄像头或播放视频'})

    if not camera.isOpened():
        return jsonify({'success': False, 'message': '摄像头未就绪'})

    if detector is None:
        return jsonify({'success': False, 'message': '检测器未初始化'})

    is_detecting = True
    frame_counter = 0
    add_log('开始检测')
    return jsonify({'success': True})


@app.route('/api/stop_detection', methods=['POST'])
@login_required
def stop_detection():
    """停止检测"""
    global is_detecting

    is_detecting = False
    add_log('停止检测')
    return jsonify({'success': True})


@app.route('/api/upload_file', methods=['POST'])
@login_required
def upload_file():
    """上传文件并控制播放"""
    global camera, is_detecting, video_file_path, video_thread_running

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})

    action = request.form.get('action', 'upload')
    print(f"文件上传操作: {action}, 文件名: {file.filename}")

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{upload_folder}/{timestamp}_{file.filename}'
    file.save(filename)
    print(f"文件已保存: {filename}")

    if action == 'play':
        video_thread_running = False
        time.sleep(0.5)

        if camera is not None:
            camera.release()
            camera = None
            time.sleep(0.5)

        print(f"打开视频文件: {filename}")
        camera = cv2.VideoCapture(filename)

        if camera is not None and camera.isOpened():
            fps = camera.get(cv2.CAP_PROP_FPS)
            frame_count = camera.get(cv2.CAP_PROP_FRAME_COUNT)
            width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

            print(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")

            video_file_path = filename
            is_detecting = False
            video_thread_running = True

            add_log(f'视频文件已加载: {file.filename}')
            return jsonify({
                'success': True,
                'message': '视频加载成功',
                'video_info': {
                    'fps': fps,
                    'frames': frame_count,
                    'width': width,
                    'height': height
                }
            })
        else:
            return jsonify({'success': False, 'message': '视频文件无法打开'})

    add_log(f'文件已上传: {file.filename}')
    return jsonify({'success': True, 'message': '文件上传成功'})


@app.route('/video_feed')
@login_required
def video_feed():
    """视频流路由"""
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ========== 新增：实时校准 API ==========
@app.route('/api/calibrate_perspective', methods=['POST'])
@login_required
def api_calibrate_perspective():
    """实时更新透视变换参数"""
    global perspective_transformer, speed_calculator
    
    try:
        data = request.json
        # 获取源点 (图像坐标)
        src_points = data.get('src_points')
        # 获取目标点 (现实坐标，主要修改深度/高度)
        dst_points = data.get('dst_points')
        
        if not src_points or not dst_points:
            return jsonify({'success': False, 'message': '缺少参数'})
            
        # 更新变换器
        if perspective_transformer:
            success = perspective_transformer.update_config(src_points, dst_points)
            if success:
                add_log('透视变换参数已实时更新')
                return jsonify({'success': True, 'message': '校准生效'})
            else:
                return jsonify({'success': False, 'message': '矩阵计算失败'})
        else:
            return jsonify({'success': False, 'message': '变换器未初始化'})
            
    except Exception as e:
        print(f"校准失败: {e}")
        return jsonify({'success': False, 'message': str(e)})

# ========== 图表数据API ==========
@app.route('/api/chart_data')
@login_required
def get_chart_data_api():
    """获取图表数据（简化版）"""
    with app.app_context():
        query = db.session.query(
            VehicleRecord.vehicle_type,
            VehicleRecord.speed,
            VehicleRecord.timestamp
        )

        records = query.limit(1000).all()

        speeds = [r.speed for r in records if r.speed]

        stats = {
            'total': len(records),
            'avg_speed': sum(speeds) / len(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'type_count': len(set([r.vehicle_type for r in records if r.vehicle_type]))
        }

        type_dict = {}
        for r in records:
            if r.vehicle_type:
                type_dict[r.vehicle_type] = type_dict.get(r.vehicle_type, 0) + 1

        speed_ranges = ['0-30', '30-60', '60-80', '80-100', '100+']
        speed_counts = [0, 0, 0, 0, 0]
        for s in speeds:
            if s < 30:
                speed_counts[0] += 1
            elif s < 60:
                speed_counts[1] += 1
            elif s < 80:
                speed_counts[2] += 1
            elif s < 100:
                speed_counts[3] += 1
            else:
                speed_counts[4] += 1

        recent = []
        for i, record in enumerate(sorted(records, key=lambda x: x.timestamp, reverse=True)[:50]):
            recent.append({
                'id': i + 1,
                'vehicle_id': f'vehicle_{i}',
                'speed': record.speed if record.speed else 0,
                'vehicle_type': record.vehicle_type or '未知',
                'timestamp': record.timestamp.strftime(
                    '%Y-%m-%d %H:%M:%S') if record.timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        return jsonify({
            'stats': stats,
            'vehicle_types': {
                'labels': list(type_dict.keys()) or ['car', 'bus', 'truck'],
                'values': list(type_dict.values()) or [1, 1, 1]
            },
            'speed_distribution': {
                'labels': speed_ranges,
                'values': speed_counts
            },
            'recent_records': recent
        })


# ========== 系统设置API ==========
system_settings = {
    'confidence': 0.5,
    'max_distance': 100,
    'fps': 30,
    # 'pixels_to_meters': 0.05, # 已废弃
    'data_retention': 30,
    'auto_backup': True
}


@app.route('/api/settings', methods=['GET'])
@login_required
def api_get_settings():
    """获取系统设置"""
    return jsonify(system_settings)


@app.route('/api/settings', methods=['POST'])
@login_required
def api_save_settings():
    """保存系统设置"""
    global system_settings
    data = request.json
    system_settings.update(data)
    add_log('系统设置已更新')
    return jsonify({'success': True})


@app.route('/api/db_stats')
@login_required
def api_get_db_stats():
    """获取数据库统计"""
    with app.app_context():
        return jsonify({
            'records': VehicleRecord.query.count(),
            'stats': SpeedStats.query.count(),
            'logs': DetectionLog.query.count()
        })


@app.route('/api/backup_database', methods=['POST'])
@login_required
def api_backup_database():
    """备份数据库"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'backups/backup_{timestamp}.sql'
        os.makedirs('backups', exist_ok=True)

        add_log(f'数据库备份: {filename}')
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/clean_old_data', methods=['POST'])
@login_required
def api_clean_old_data():
    """清理过期数据"""
    try:
        days = system_settings.get('data_retention', 30)
        cutoff = datetime.now() - timedelta(days=days)

        with app.app_context():
            deleted_records = VehicleRecord.query.filter(VehicleRecord.timestamp < cutoff).delete()
            deleted_stats = SpeedStats.query.filter(SpeedStats.timestamp < cutoff).delete()
            deleted_logs = DetectionLog.query.filter(DetectionLog.timestamp < cutoff).delete()
            db.session.commit()

        add_log(
            f'清理 {days} 天前的数据: 删除了 {deleted_records} 条记录, {deleted_stats} 条统计, {deleted_logs} 条日志')
        return jsonify(
            {'success': True, 'message': f'清理完成: 记录{deleted_records}, 统计{deleted_stats}, 日志{deleted_logs}'})
    except Exception as e:
        print(f"清理数据失败: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/clear_data', methods=['POST'])
@login_required
def api_clear_data():
    """清空所有数据"""
    try:
        with app.app_context():
            num_records = VehicleRecord.query.delete()
            num_stats = SpeedStats.query.delete()
            num_logs = DetectionLog.query.delete()
            db.session.commit()

            add_log(f'清空数据: 删除了 {num_records} 条车辆记录, {num_stats} 条速度统计, {num_logs} 条日志')
            return jsonify({
                'success': True,
                'message': f'数据已清空 (车辆记录: {num_records}, 速度统计: {num_stats}, 日志: {num_logs})'
            })
    except Exception as e:
        print(f"清空数据失败: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/export_report')
@login_required
def api_export_report():
    """导出报表"""
    range_type = request.args.get('range', 'all')

    with app.app_context():
        records = VehicleRecord.query.order_by(VehicleRecord.timestamp.desc()).limit(1000).all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', '车辆ID', '速度(km/h)', '类型', '时间'])

        for r in records:
            writer.writerow([r.id, r.vehicle_id, r.speed, r.vehicle_type, r.timestamp])

        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=vehicle_records_{range_type}.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response


# ========== 主程序 ==========
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('backups', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)