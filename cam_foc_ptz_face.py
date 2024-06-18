import cv2
import threading
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer


HD_1080P = {"WIDTH": 1920, "HEIGHT": 1080}
HD_720P = {"WIDTH": 1280, "HEIGHT": 720}
HD_360P = {"WIDTH": 640, "HEIGHT": 360}

import time
from Detector.yunet_detector import YuNetDet
from Tracker.deepsort_tracker import DeepSort
from mushfoc.MushFOC import MushFOC
from servo.SimplePID import SimplePID

class MushPTZ:

    address = None
    bus = None
    
    yaw_offset = 0
    pitch_offset = 0
    yaw_max_angle = 0
    pitch_max_angle = 0
    yaw_min_angle = 0
    pitch_min_angle = 0
    current_yaw_angle = 0
    current_pitch_angle = 0

    def __init__(self, yaw_address=0x51, pitch_address=0x52, bus=1):
        self.yaw_foc = MushFOC(address=yaw_address, bus=bus)
        self.pitch_foc = MushFOC(address=pitch_address, bus=bus)
        self.current_yaw_angle = 0
        self.current_pitch_angle = 0
        # 暂时使用PID控制，后续可以考虑使用卡尔曼滤波
        self.yaw_pid_g = SimplePID(0, -10, 10, 0.15, 0.02, 0.00, 20)
        self.pitch_pid_g = SimplePID(0, -10, 10, 0.15, 0.02, 0.00, 20)
        
    def set_offset(self, yaw_offset=0, pitch_offset=0):
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset
        
    def set_angle_range(self, yaw_max_angle=90, pitch_max_angle=90, yaw_min_angle=-90, pitch_min_angle=-90):
        if(yaw_max_angle<yaw_min_angle):
            print(f"yaw_max_angle:{yaw_max_angle} should be larger than yaw_min_angle:{yaw_min_angle}")
        else:
            self.yaw_max_angle = yaw_max_angle
            self.pitch_max_angle = pitch_max_angle
        if(pitch_max_angle<pitch_min_angle):
            print(f"pitch_max_angle:{pitch_max_angle} should be larger than pitch_min_angle:{pitch_min_angle}")
        else:
            self.yaw_min_angle = yaw_min_angle
            self.pitch_min_angle = pitch_min_angle

    def move(self, yaw_angle, pitch_angle):
        # 计算相对偏移的角度
        yaw_angle1 = self.yaw_pid_g .get_output_value(yaw_angle)
        pitch_angle1 = self.pitch_pid_g.get_output_value(pitch_angle)
        # 更新当前角度
        self.current_yaw_angle -= yaw_angle1
        self.current_pitch_angle -= pitch_angle1
        # 限制角度范围
        self.current_yaw_angle = max(min(self.current_yaw_angle, self.yaw_max_angle), self.yaw_min_angle)
        self.current_pitch_angle = max(min(self.current_pitch_angle, self.pitch_max_angle), self.pitch_min_angle)
        # 设置云台角度
        self.yaw_foc.set_angle(int(self.current_yaw_angle+self.yaw_offset))
        self.pitch_foc.set_angle(int(self.current_pitch_angle+self.pitch_offset))
        # 返回实际相对偏移的角度
        return yaw_angle1, pitch_angle1
        
        
    def start(self):
        self.yaw_foc.set_on()
        self.pitch_foc.set_on()
    
    def stop(self):
        self.yaw_foc.set_off()
        self.pitch_foc.set_off()
        
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

        self.ptz = MushPTZ()
        self.ptz.set_offset(yaw_offset=30, pitch_offset=50)
        self.ptz.set_angle_range(yaw_max_angle=90, pitch_max_angle=90, yaw_min_angle=-90, pitch_min_angle=-90)
        self.ptz.move(0, 0)
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
        self.ptz.move(0, 0)
        while self.running:
            time.sleep(0.05)
            dx_move = self.dx/1280./2.
            dy_move = self.dy/720./2.
            
            # dx_move dy_move 为相对于屏幕中心的偏移量
            d_x, d_y = self.ptz.move(dx_move*90, dy_move*-40)
            # soft fix
            self.dx += d_x*8
            self.dy += d_y*8
            # print(f'moving: {self.dx:.2f}, {self.dy:.2f}, {d_x:.2f}, {d_y:.2f}')
            
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
        self.ptz.move(0, 0)

    @staticmethod
    def find_camera_index():
        max_index_to_check = 10
        for index in range(max_index_to_check):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                cap.release()
                return index
        raise ValueError("No camera found.")
    
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
                img0 = cv2.rectangle(img0, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 255, 255), thickness=-1)
                img0 = cv2.putText(img0, str(id) , (int(ltrb[0]), int(ltrb[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                min_id = id
            else :
                img0 = cv2.rectangle(img0, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), thickness=-1)
                img0 = cv2.putText(img0, str(id) , (int(ltrb[0]), int(ltrb[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return img0
        
class AIBooster:
    
    
    def __init__(self):
        self.detector = YuNetDet(platform='rknn-3566')
        self.tracker = DeepSort(max_age=5, embedder='sface_rknn-3566')
        self.prev_time = 0
    
    def get_tracks(self):
        return self.tracks
    
    def detach_camera(self):
        self.camera = None
        self.running = False
        self.thread_cam.join()
    
    def infer_frame_with_vis(self, image):
        # cv2.imwrite('./raw_image.jpg', image)
        image_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 计算帧率
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        # 模型推理
        bbs = self.detector.inference(image_tmp)
        infer_time = time.time()
        self.tracks = self.tracker.update_tracks(bbs, frame=image_tmp)
        sort_time = time.time()
        print(f"fps:{fps:.2f} infer:{infer_time-current_time:.2f} sort:{sort_time-infer_time:.2f} dectect:{len(bbs)} tracks:{len(self.tracks)} ")
        
        return self.tracks

    @staticmethod
    def img2bytes(image):
        return bytes(cv2.imencode('.jpg', image)[1])


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/live':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                # 从视频流中读取帧。
                time.sleep(0.001)
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
                        <img src="/live">
                    </body>
                </html>
                """
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

resolution=HD_720P

def run_server():
    server.serve_forever(0.01)

try:
    camera = Camera(resolution=resolution)
    print("Camera initialized")
    model = AIBooster()

    server = HTTPServer(('0.0.0.0', 2233), VideoStreamHandler)
    print("Server started")
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    prev_time = time.time()
    while True:
        
        time.sleep(0.001)
        frame = camera.get_frame()
        # infer_time = current_time
        if frame is not None:
            tracks = model.infer_frame_with_vis(frame)
            # infer_time = time.time()
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
                break

except KeyboardInterrupt:
    server.shutdown()    # 停止服务器
    camera.release()
    print("Camera released")
    