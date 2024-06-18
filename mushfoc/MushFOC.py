
import smbus
import time

class MushFOC:

    # contrl CMD
    FOC_ERROR                   = 0x00
    FOC_OK                      = 0x01
    FOC_INIT_DONE               = 0x02
    
    FOC_SET_ON                  = 0x20
    FOC_SET_OFF                 = 0x21
    
    FOC_SET_ANGLE               = 0x30
    FOC_GET_ANGLE               = 0x31
    FOC_SET_MAX_OUTPUT_ANGLE    = 0x32
    FOC_GET_MAX_OUTPUT_ANGLE    = 0x33
    FOC_SET_MAX_ZERO_ANGLE      = 0x34
    FOC_GET_MAX_ZERO_ANGLE      = 0x35
    FOC_SET_MIN_ZERO_ANGLE      = 0x36
    FOC_GET_MIN_ZERO_ANGLE      = 0x37
    
    FOC_SET_VELOCITY            = 0x40
    FOC_GET_VELOCITY            = 0x41
    FOC_SET_MAX_OUTPUT_VELOCITY = 0x42
    FOC_GET_MAX_OUTPUT_VELOCITY = 0x43
    
    FOC_SET_CURRENT             = 0x50
    FOC_GET_CURRENT             = 0x51
    FOC_SET_MAX_OUTPUT_CURRENT  = 0x52
    FOC_GET_MAX_OUTPUT_CURRENT  = 0x53
    
    FOC_SET_ACCUMULATED_THRESHOLD_ROATIO = 0x60
    
    FOC_RESTART = 0xA5

    address = None
    bus = None

    def __init__(self, address=0x51, bus=1, restart=False):
        self.address = address
        self.bus = smbus.SMBus(bus)
        if restart:
            self.restart()
            time.sleep(5)
        
    def __del__(self):
        # self.set_off()
        self.bus.close()
        
    def restart(self):
        """ Restarts the FOC.
        """
        self.bus.write_i2c_block_data(self.address, self.FOC_RESTART, [0x00, 0x00])
        
    def set_on(self):
        """ Sets the FOC Power on.
        """
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_ON, [0x00, 0x00])
        
    def set_off(self):
        """ Sets the FOC Power off.
        """
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_OFF, [0x00, 0x00])
        
    def set_angle(self, angle:int, wait_ms=1):
        """ Sets the angle.

        Args:
            angle (int): -90 means -90 degree, 0 means 0 degree, 90 means 90 degree
            wait_ms (int, optional): _description_. Defaults to 10.
        """
        angle = angle + 180
        angle_H = angle>>8
        angle_L = angle&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_ANGLE, [angle_H, angle_L])
        time.sleep(wait_ms/1000)
    
    def get_angle(self):
        pass
        
    def set_max_output_angle(self, angle:int):
        """ Sets the max output angle in PID control.

        Args:
            angle (int): -90 means -90 degree, 0 means 0 degree, 90 means 90 degree
        """
        angle = angle + 180
        angle_H = angle>>8
        angle_L = angle&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_MAX_OUTPUT_ANGLE, [angle_H, angle_L])
    
    def get_max_output_angle(self):
        pass
    
    def set_max_zero_angle(self, angle:int):
        """ Sets the max zero angle.

        Args:
            angle (int): -90 means -90 degree, 0 means 0 degree, 90 means 90 degree
        """
        angle = angle + 180
        angle_H = angle>>8
        angle_L = angle&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_MAX_ZERO_ANGLE, [angle_H, angle_L])
        
    def get_max_zero_angle(self):
        pass
    
    def set_min_zero_angle(self, angle:int):
        """ Sets the min zero angle.

        Args:
            angle (int): -90 means -90 degree, 0 means 0 degree, 90 means 90 degree
        """
        angle = angle + 180
        angle_H = angle>>8
        angle_L = angle&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_MIN_ZERO_ANGLE, [angle_H, angle_L])
        
    def get_min_zero_angle(self):
        pass    
        
    def set_velocity(self, velocity:int):
        """ Sets the velocity.

        Args:
            velocity (int): 1000 means 1.000 rad/s, 100 means 0.100 rad/s, 1 means 0.001 rad/s
        """
        velocity_H = velocity>>8
        velocity_L = velocity&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_VELOCITY, [velocity_H, velocity_L])
    
    def get_velocity(self):
        pass
    
    def set_max_output_velocity(self, velocity:int):
        """ Sets the max output velocity in PID control.

        Args:
            velocity (int): 1000 means 1.000 rad/s, 100 means 0.100 rad/s, 1 means 0.001 rad/s
        """
        velocity_H = velocity>>8
        velocity_L = velocity&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_MAX_OUTPUT_VELOCITY, [velocity_H, velocity_L])
        
    def get_max_output_velocity(self):
        pass
    
    def set_current(self, current:int):
        """ Sets the current.

        Args:
            current (int): 1000 means 1000mA, 100 means 100mA, 1 means 1mA
        """
        current_H = current>>8
        current_L = current&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_CURRENT, [current_H, current_L])
        
    def get_current(self):
        pass
    
    def set_max_output_current(self, current:int):
        """ Sets the max output current in PID control.

        Args:
            current (int): 1000 means 1000mA, 100 means 100mA, 1 means 1mA
        """
        current_H = current>>8
        current_L = current&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_MAX_OUTPUT_CURRENT, [current_H, current_L])
        
    def get_max_output_current(self):
        pass
    
    
    def set_accumulated_threshold_ratio(self, ratio:int):
        """ Sets the accumulated threshold ratio.
        
        Args:
            ratio (int): 1000 means 1.000, 100 means 0.100, 1 means 0.001
        """
        ratio_H = ratio>>8
        ratio_L = ratio&0xFF
        self.bus.write_i2c_block_data(self.address, self.FOC_SET_ACCUMULATED_THRESHOLD_ROATIO, [ratio_H, ratio_L])
        
    def get_accumulated_threshold_ratio(self):
        pass
    

foc = MushFOC(address=0x52)
foc.set_angle(30)
