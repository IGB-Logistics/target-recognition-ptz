import sys
sys.path.append('..')

import time
import math
from MushFOC import MushFOC
from Ext.SSD1306 import SSD1306, Canvas
from Ext.MPU6050 import MPU6050
from Ext.Kalman import KalmanAngle
from PIL import ImageFont, ImageDraw

class MushPTZ:

    address = None
    bus = None
    
    yaw_offset = 0
    pitch_offset = 0
    yaw_max_angle = 0
    pitch_max_angle = 0
    yaw_min_angle = 0
    pitch_min_angle = 0

    def __init__(self, yaw_address=0x51, pitch_address=0x52, bus=1):
        self.yaw_foc = MushFOC(address=yaw_address, bus=bus)
        self.pitch_foc = MushFOC(address=pitch_address, bus=bus)
        
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
        _yaw_angle = max(self.yaw_min_angle, yaw_angle)
        _yaw_angle = min(self.yaw_max_angle, yaw_angle)
        _pitch_angle = max(self.pitch_min_angle, pitch_angle)
        _pitch_angle = min(self.pitch_max_angle, pitch_angle)
        
        self.yaw_foc.set_angle(int(_yaw_angle + self.yaw_offset))
        self.pitch_foc.set_angle(int(_pitch_angle + self.pitch_offset))
        
    def start(self):
        self.yaw_foc.set_on()
        self.pitch_foc.set_on()
    
    def stop(self):
        self.yaw_foc.set_off()
        self.pitch_foc.set_off()
        
if __name__ == "__main__":

    monitor = SSD1306(port=1, address=0x3C)
    font = ImageFont.load_default()
    with Canvas(monitor) as draw:
        draw.text((0, 0), 'hello', font=font, fill=255)
        
    ptz = MushPTZ(yaw_address=0x51, pitch_address=0x52, bus=1)
    mpu = MPU6050(address=0x69)
    kalmanYaw = KalmanAngle()
    kalmanPitch = KalmanAngle()
    kalmanRoll = KalmanAngle()
    pre_time = time.time()
    
    while(True):
        dt = time.time() - pre_time
        pre_time = time.time()
        
        accel_data = mpu.get_accel_data()
        gyro_data = mpu.get_gyro_data()

        angleYaw = math.atan2(accel_data['x'], accel_data['y']) * 180 / math.pi
        gyroGx  = -gyro_data['x']
        yaw = kalmanYaw.run(angleYaw, gyroGx, dt)

        anglePitch = math.atan2(-accel_data['x'], accel_data['z']) * 180 / math.pi
        gyroGy  = gyro_data['y']
        pitch = kalmanPitch.run(anglePitch, gyroGy, dt)

        angleRoll = math.atan2(accel_data['y'], accel_data['z']) * 180 / math.pi
        gyroGz  = -gyro_data['z']
        roll = kalmanRoll.run(angleRoll, gyroGz, dt)
        
        # draw.text((0, 16), str(yaw), font=font, fill=255)
        # draw.text((0, 24), str(pitch), font=font, fill=255)
        # draw.text((0, 24), str(roll), font=font, fill=255)

        # tx = gyro_data['x']
        # ty = gyro_data['y']
        # tz = gyro_data['z']
        # print(f'{tx:3.2f} {ty:3.2f} {tz:3.2f}')
        print(f'{angleYaw:3.2f} {anglePitch:3.2f} {angleRoll:3.2f}')
        # print(f'{yaw:3.2f} {pitch:3.2f} {roll:3.2f}')

        time.sleep(0.5)