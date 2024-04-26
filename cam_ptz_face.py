import cv2
import threading
import numpy as np
import torch
from ais_bench.infer.interface import InferSession
import yunet_det_utils
from http.server import BaseHTTPRequestHandler, HTTPServer


HD_1080P = {"WIDTH": 1920, "HEIGHT": 1080}
HD_720P = {"WIDTH": 1280, "HEIGHT": 720}
HD_360P = {"WIDTH": 640, "HEIGHT": 360}

import time
import smbus2 as smbus
from servo.PCA9685 import PCA9685
from servo.ServoPCA9685 import ServoPCA9685
from servo.ServoPCA9685 import map
from servo.SimplePID import SimplePID

from deep_sort_lite.deepsort_tracker import DeepSort

class ServoPTZ:
    YAW_ANGLE0 = 85
    PATCH_ANGLE0 = 85
    
    YAW_ANGLE_MIN = 45
    YAW_ANGLE_MAX = 135
    PATCH_ANGLE_MIN = 45
    PATCH_ANGLE_MAX = 135
    
    def __init__(self, bus=7, yaw_channel=PCA9685.CHANNEL06, pitch_channel=PCA9685.CHANNEL07):
        i2cBus = smbus.SMBus(bus)
        pca9685 = PCA9685(i2cBus)

        self.servo_yaw =   ServoPCA9685(pca9685, yaw_channel, servo_pwm = [205,409], servo_dgr=[self.YAW_ANGLE_MIN, self.YAW_ANGLE_MAX])
        self.servo_pitch = ServoPCA9685(pca9685, pitch_channel,  servo_pwm = [205,409], servo_dgr=[self.PATCH_ANGLE_MIN, self.PATCH_ANGLE_MAX])
        
        self.yaw_pid_g = SimplePID(0, -10, 10, 0.1, 0.01, 0.001, 20)
        self.pitch_pid_g = SimplePID(0, -10, 10, 0.1, 0.01, 0.001, 20)
        
        self.current_yaw_angle = self.YAW_ANGLE0
        self.current_pitch_angle = self.PATCH_ANGLE0
        self.servo_yaw.set_angle(self.current_yaw_angle)
        self.servo_pitch.set_angle(self.current_pitch_angle)
        
    def move_with_pid(self, yaw_angle, pitch_angle, cost_time = 1, step=50):
        # reset to 0
        yaw_angle += self.YAW_ANGLE0
        pitch_angle += self.PATCH_ANGLE0
        yaw_pid = SimplePID(0, self.YAW_ANGLE_MIN - self.YAW_ANGLE0, self.YAW_ANGLE_MAX - self.YAW_ANGLE0,          0.02, 0.01, 0.0)
        pitch_pid = SimplePID(0, self.YAW_ANGLE_MIN - self.PATCH_ANGLE0, self.YAW_ANGLE_MAX - self.PATCH_ANGLE0,    0.02, 0.01, 0.0)
        intvert_ms  = cost_time / step
        for u in range(step):
            yaw_angle1 = yaw_pid.get_output_value(yaw_angle - self.current_yaw_angle)
            pitch_angle1 = pitch_pid.get_output_value(pitch_angle - self.current_pitch_angle)
            self.current_yaw_angle -= yaw_angle1
            self.current_pitch_angle -= pitch_angle1
            self.servo_yaw.set_angle(self.current_yaw_angle)
            self.servo_pitch.set_angle(self.current_pitch_angle)
            # print(self.current_yaw_angle, self.current_pitch_angle)
            time.sleep(intvert_ms)

    def move_one_shot(self, yaw_angle, pitch_angle):
        # reset to 0
        yaw_angle1 = self.yaw_pid_g .get_output_value(yaw_angle)
        pitch_angle1 = self.pitch_pid_g.get_output_value(pitch_angle)
        self.current_yaw_angle -= yaw_angle1
        self.current_pitch_angle -= pitch_angle1
        self.current_yaw_angle = max(self.YAW_ANGLE_MIN, min(self.YAW_ANGLE_MAX, self.current_yaw_angle))
        self.current_pitch_angle = max(self.PATCH_ANGLE_MIN, min(self.PATCH_ANGLE_MAX, self.current_pitch_angle))
        self.servo_yaw.set_angle(self.current_yaw_angle)
        self.servo_pitch.set_angle(self.current_pitch_angle)
        # print(f"{yaw_angle1:.2f},{pitch_angle1:2f}, {self.current_yaw_angle:.2f},{self.current_pitch_angle:.2f}")
        # 1 second 50 step
        time.sleep(0.01)
        return yaw_angle1, pitch_angle1

class Camera:
    def __init__(self, resolution=HD_720P):
        self.resolution = resolution
        self.camera_index = self.find_camera_index()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.configure_camera()
        self.frame = None
        self.running = True
        self.thread_cam = threading.Thread(target=self.update, args=())
        self.thread_cam.start()
        self.prev_time = 0

        self.ptz = ServoPTZ()
        self.thread_servo = threading.Thread(target=self.move, args=())
        self.thread_servo.start()
        self.dx = 0
        self.dy = 0

    def configure_camera(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution['WIDTH'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution['HEIGHT'])

    def update(self):
        prev_time = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # frame = cv2.flip(frame, -1)

                # 计算帧率
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # 将帧率和分辨率信息绘制到图像上
                # print(f"fps:{fps:.2f}")
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Resolution: {self.resolution['WIDTH']}x{self.resolution['HEIGHT']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
         
                self.frame = frame

    def move(self):
        self.ptz.move_with_pid(0, 0)
        while self.running:
            # time.sleep(0.01)
            dx_move = self.dx/1280./2.
            dy_move = self.dy/720./2.
            
            d_x, d_y = self.ptz.move_one_shot(dx_move*90, dy_move*90)
            # soft fix
            self.dx += d_x*8
            self.dy += d_y*8
            # print(f'moving: {self.dx:.2f}, {self.dy:.2f}, {d_x:.2f}, {d_y:.2f}')
    
    def move_to_target(self):
        dx_move = self.dx/1280./2.
        dy_move = self.dy/720./2.
        self.ptz.move_with_pid(dx_move*90, dy_move*90)
        # print(f'moving: {self.dx:.2f}, {self.dy:.2f}')
            
    def get_frame(self):
        return self.frame
    
    def update_dxy(self, dx, dy):
        self.dx, self.dy = dx, dy
        # print(f'update: {self.dx:.2f}, {self.dy:.2f}')

    def release(self):
        self.running = False
        self.thread_cam.join()
        self.thread_servo.join()
        self.cap.release()
        self.ptz.move_with_pid(0, 0)

    @staticmethod
    def find_camera_index():
        max_index_to_check = 10
        for index in range(max_index_to_check):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                cap.release()
                return index
        raise ValueError("No camera found.")
    
    def draw_bbox(self, bbox, img0, color, wt, names):
        """在图片上画预测框"""
        det_result_str = ''
        
        screen_center_x = int(img0.shape[1] / 2)
        screen_center_y = int(img0.shape[0] / 2)
        min_dis = 100000
        min_idx = -1
        min_center_x = screen_center_x
        min_center_y = screen_center_y
        center_x = screen_center_x
        center_y = screen_center_y
        
        for idx, class_id in enumerate(bbox[:, 5]):
            if names[int(class_id)] != "person" or float(bbox[idx][4] < float(0.05)):
                continue
            img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                                 color, wt)
            img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            det_result_str += '{} {} {} {} {} {}\n'.format(
                names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    
            center_x = int((int(bbox[idx][0]) + int(bbox[idx][2])) / 2)
            center_y = int((int(bbox[idx][1]) + int(bbox[idx][3])) / 2)
            
            offset_x = center_x - screen_center_x
            offset_y = center_y - screen_center_y
            dis = abs(offset_x) + abs(offset_y)
            if dis < min_dis:
               min_dis = dis
               min_center_x = center_x
               min_center_y = center_y
               min_idx = idx

        # print("方格中心偏移：X轴偏移={:.2f}, Y轴偏移={:.2f}".format(center_x-min_center_x, center_y-min_center_y))
        if min_idx != -1:
            self.dx = screen_center_x-min_center_x
            self.dy = min_center_y-screen_center_y
        
            img0 = cv2.putText(img0, 'dx={:.2f}'.format(self.dx), (int(bbox[min_idx][0]), int(bbox[min_idx][1] + 48)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            img0 = cv2.putText(img0, 'dy={:.2f}'.format(self.dy), (int(bbox[min_idx][0]), int(bbox[min_idx][1] + 64)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        img0 = cv2.arrowedLine(img0, (screen_center_x, screen_center_y), (min_center_x, min_center_y), (0, 0, 255), 2)
    
        return img0


    def draw_tracks_box(self, tracks, img0):
        """在图片上画预测框"""
        screen_center_x = int(img0.shape[1] / 2)
        screen_center_y = int(img0.shape[0] / 2)
        min_id = -1
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            id = track.track_id
            ltrb = track.to_ltrb()
            if min_id == -1 :
                img0 = cv2.rectangle(img0, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 4)
                img0 = cv2.putText(img0, str(id) , (int(ltrb[0]), int(ltrb[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                min_id = id
            else :
                img0 = cv2.rectangle(img0, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 1)
                img0 = cv2.putText(img0, str(id) , (int(ltrb[0]), int(ltrb[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return img0

    def track_bbox(self, bbox, img0, color, wt, names):
        """在图片上画预测框"""
        det_result_str = ''
        
        screen_center_x = int(img0.shape[1] / 2)
        screen_center_y = int(img0.shape[0] / 2)
        min_dis = 100000
        min_idx = -1
        min_center_x = screen_center_x
        min_center_y = screen_center_y
        center_x = screen_center_x
        center_y = screen_center_y
        
        
class AIBooster:
    YUNET_SIZE_W = 160
    YUNET_SIZE_H = 120
    
    
    def __init__(self, model_path):
        self.model = yunet_det_utils.YuNetDet(model_path)
        self.tracker = DeepSort(max_age=5, embedder='sface_npu')
        self.prev_time = 0
    
    def get_tracks(self):
        return self.tracks
    
    def detach_camera(self):
        self.camera = None
        self.running = False
        self.thread_cam.join()
    
    def infer_frame_with_vis(self, image, bgr2rgb=True):        
        # 计算帧率
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # 模型推理
        bboxes, landmarks, scores = self.model.inference(image)
        infer_time = time.time()
        
        # DeepSort
        # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        boxes = bboxes
        confidence = scores
        detection_class = np.ones(len(boxes))
        # boxes_xywh = [[box[0], box[1], box[2], box[3]] for box in boxes]
        boxes_xywh = [[image.shape[1] * box[0] / self.YUNET_SIZE_W, 
                       image.shape[0] * box[1] / self.YUNET_SIZE_H, 
                       image.shape[1] * box[2] / self.YUNET_SIZE_W,
                       image.shape[0] * box[3] / self.YUNET_SIZE_H] for box in boxes]
        bbs = list(zip(boxes_xywh, confidence, detection_class))
        self.tracks = self.tracker.update_tracks(bbs, frame=image)
        sort_time = time.time()
        # print(f"fps:{fps:.2f} infer:{infer_time-current_time:.2f} sort:{sort_time-infer_time:.2f} dectect:{len(bboxes)} tracks:{len(self.tracks)} ")
        
        return self.tracks

    @staticmethod
    def img2bytes(image):
        return bytes(cv2.imencode('.jpg', image)[1])

# 定义一个辅助函数，用于从RTSP源读取视频流。
def generate_frames():
    while True:
        # 从视频流中读取帧。
        frame = camera.get_frame()
        frame = camera.draw_bbox(model.yolo_pred, frame, (0, 255, 0), 2, model.labels_dict)
        # 将帧转换为JPEG格式。
        ret, buffer = cv2.imencode(".jpg", frame)
        # 将JPEG数据转换为字节字符串，并将其作为流响应返回。
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                # 从视频流中读取帧。
                time.sleep(0.01)
                frame = camera.get_frame()
                if frame is not None:
                    # frame = camera.draw_bbox(model.yolo_pred, frame, (0, 255, 0), 2, model.labels_dict)
                    frame = camera.draw_tracks_box(model.tracks, frame)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
        if self.path == '/cam':
            self.send_response(200)
            html = """
                <html>
                    <head>
                        <title>USB Camera Streaming</title>
                    </head>
                    <body>
                        <h1>USB Camera Streaming</h1>
                        <img src="/video_feed">
                    </body>
                </html>
                """
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

model_path = './model/yunet_120x160.om'
resolution=HD_720P

def run_server():
    server.serve_forever(0.01)

try:
    model = AIBooster(model_path)
    camera = Camera(resolution=resolution)
    print("Camera initialized")

    server = HTTPServer(('0.0.0.0', 2233), VideoStreamHandler)
    print("Server started")
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    while True:
        time.sleep(0.001)
        frame = camera.get_frame()
        if frame is not None:
            tracks = model.infer_frame_with_vis(frame)
            # 更新dx dy 
            screen_center_x = int(frame.shape[1] / 2)
            screen_center_y = int(frame.shape[0] / 2)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                dx = screen_center_x - (ltrb[0]+ltrb[2])/2
                dy =  (ltrb[1]+ltrb[3])/2 - screen_center_y
                camera.update_dxy(dx, dy)
                # camera.move_to_target()
                break

except KeyboardInterrupt:
    server.shutdown()    # 停止服务器
    camera.release()
    print("Camera released")
    