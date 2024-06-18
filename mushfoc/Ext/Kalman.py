import numpy as np

class KalmanAngle:
    def __init__(self):
        # 采集来的角度和角速度  
        self.angle = 0.0
        self.angle_dot = 0.0
        self.angle_0 = 0.0
        self.angle_dot_0 = 0.0

        self.q_bias = 0.0
        self.angle_err = 0.0
        self.PCt_0 = 0.0
        self.PCt_1 = 0.0
        self.E = 0.0
        self.K_0 = 0.0
        self.K_1 = 0.0
        self.t_0 = 0.0
        self.t_1 = 0.0
        
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.Pdot = np.array([0.0, 0.0, 0.0, 0.0])
        # 角度数据置信度,角速度数据置信度  
        self.Q_angle = 0.001
        self.Q_gyro = 0.005
        self.R_angle = 0.5
        self.C_0 = 0.1

    def run(self, angle_m, gyro_m, dt):
        self.angle += (gyro_m - self.q_bias) * dt
        self.angle_err = angle_m - self.angle
        self.Pdot[0] = self.Q_angle - self.P[0][1] - self.P[1][0]
        self.Pdot[1] = -self.P[1][1]
        self.Pdot[2] = -self.P[1][1]
        self.Pdot[3] = self.Q_gyro
        self.P[0][0] += self.Pdot[0] * dt
        self.P[0][1] += self.Pdot[1] * dt
        self.P[1][0] += self.Pdot[2] * dt
        self.P[1][1] += self.Pdot[3] * dt
        self.PCt_0 = self.C_0 * self.P[0][0]
        self.PCt_1 = self.C_0 * self.P[1][0]
        self.E = self.R_angle + self.C_0 * self.PCt_0
        self.K_0 = self.PCt_0 / self.E
        self.K_1 = self.PCt_1 / self.E
        self.t_0 = self.PCt_0
        self.t_1 = self.C_0 * self.P[0][1]
        self.P[0][0] -= self.K_0 * self.t_0
        self.P[0][1] -= self.K_0 * self.t_1
        self.P[1][0] -= self.K_1 * self.t_0
        self.P[1][1] -= self.K_1 * self.t_1
        self.angle += self.K_0 * self.angle_err
        self.q_bias += self.K_1 * self.angle_err
        self.angle_dot = gyro_m - self.q_bias
        return self.angle