"""ICM20948 sensor controller."""

from __future__ import annotations

import importlib
import importlib.util
import math
import time

true = 0x01
false = 0x00

# define ICM-20948 Device I2C address
I2C_ADD_ICM20948 = 0x68
I2C_ADD_ICM20948_AK09916 = 0x0C
I2C_ADD_ICM20948_AK09916_READ = 0x80
I2C_ADD_ICM20948_AK09916_WRITE = 0x00
# define ICM-20948 Register
# user bank 0 register
REG_ADD_WIA = 0x00
REG_VAL_WIA = 0xEA
REG_ADD_USER_CTRL = 0x03
REG_VAL_BIT_DMP_EN = 0x80
REG_VAL_BIT_FIFO_EN = 0x40
REG_VAL_BIT_I2C_MST_EN = 0x20
REG_VAL_BIT_I2C_IF_DIS = 0x10
REG_VAL_BIT_DMP_RST = 0x08
REG_VAL_BIT_DIAMOND_DMP_RST = 0x04
REG_ADD_PWR_MIGMT_1 = 0x06
REG_VAL_ALL_RGE_RESET = 0x80
REG_VAL_RUN_MODE = 0x01  # Non low-power mode
REG_ADD_LP_CONFIG = 0x05
REG_ADD_PWR_MGMT_1 = 0x06
REG_ADD_PWR_MGMT_2 = 0x07
REG_ADD_ACCEL_XOUT_H = 0x2D
REG_ADD_ACCEL_XOUT_L = 0x2E
REG_ADD_ACCEL_YOUT_H = 0x2F
REG_ADD_ACCEL_YOUT_L = 0x30
REG_ADD_ACCEL_ZOUT_H = 0x31
REG_ADD_ACCEL_ZOUT_L = 0x32
REG_ADD_GYRO_XOUT_H = 0x33
REG_ADD_GYRO_XOUT_L = 0x34
REG_ADD_GYRO_YOUT_H = 0x35
REG_ADD_GYRO_YOUT_L = 0x36
REG_ADD_GYRO_ZOUT_H = 0x37
REG_ADD_GYRO_ZOUT_L = 0x38
REG_ADD_EXT_SENS_DATA_00 = 0x3B
REG_ADD_REG_BANK_SEL = 0x7F
REG_VAL_REG_BANK_0 = 0x00
REG_VAL_REG_BANK_1 = 0x10
REG_VAL_REG_BANK_2 = 0x20
REG_VAL_REG_BANK_3 = 0x30

# user bank 1 register
# user bank 2 register
REG_ADD_GYRO_SMPLRT_DIV = 0x00
REG_ADD_GYRO_CONFIG_1 = 0x01
REG_VAL_BIT_GYRO_DLPCFG_2 = 0x10  # bit[5:3]
REG_VAL_BIT_GYRO_DLPCFG_4 = 0x20  # bit[5:3]
REG_VAL_BIT_GYRO_DLPCFG_6 = 0x30  # bit[5:3]
REG_VAL_BIT_GYRO_FS_250DPS = 0x00  # bit[2:1]
REG_VAL_BIT_GYRO_FS_500DPS = 0x02  # bit[2:1]
REG_VAL_BIT_GYRO_FS_1000DPS = 0x04  # bit[2:1]
REG_VAL_BIT_GYRO_FS_2000DPS = 0x06  # bit[2:1]
REG_VAL_BIT_GYRO_DLPF = 0x01  # bit[0]
REG_ADD_ACCEL_SMPLRT_DIV_2 = 0x11
REG_ADD_ACCEL_CONFIG = 0x14
REG_VAL_BIT_ACCEL_DLPCFG_2 = 0x10  # bit[5:3]
REG_VAL_BIT_ACCEL_DLPCFG_4 = 0x20  # bit[5:3]
REG_VAL_BIT_ACCEL_DLPCFG_6 = 0x30  # bit[5:3]
REG_VAL_BIT_ACCEL_FS_2g = 0x00  # bit[2:1]
REG_VAL_BIT_ACCEL_FS_4g = 0x02  # bit[2:1]
REG_VAL_BIT_ACCEL_FS_8g = 0x04  # bit[2:1]
REG_VAL_BIT_ACCEL_FS_16g = 0x06  # bit[2:1]
REG_VAL_BIT_ACCEL_DLPF = 0x01  # bit[0]

# user bank 3 register
REG_ADD_I2C_SLV0_ADDR = 0x03
REG_ADD_I2C_SLV0_REG = 0x04
REG_ADD_I2C_SLV0_CTRL = 0x05
REG_VAL_BIT_SLV0_EN = 0x80
REG_VAL_BIT_MASK_LEN = 0x07
REG_ADD_I2C_SLV0_DO = 0x06
REG_ADD_I2C_SLV1_ADDR = 0x07
REG_ADD_I2C_SLV1_REG = 0x08
REG_ADD_I2C_SLV1_CTRL = 0x09
REG_ADD_I2C_SLV1_DO = 0x0A

# define ICM-20948 Register  end

# define ICM-20948 MAG Register
REG_ADD_MAG_WIA1 = 0x00
REG_VAL_MAG_WIA1 = 0x48
REG_ADD_MAG_WIA2 = 0x01
REG_VAL_MAG_WIA2 = 0x09
REG_ADD_MAG_ST2 = 0x10
REG_ADD_MAG_DATA = 0x11
REG_ADD_MAG_CNTL2 = 0x31
REG_VAL_MAG_MODE_PD = 0x00
REG_VAL_MAG_MODE_SM = 0x01
REG_VAL_MAG_MODE_10HZ = 0x02
REG_VAL_MAG_MODE_20HZ = 0x04
REG_VAL_MAG_MODE_50HZ = 0x05
REG_VAL_MAG_MODE_100HZ = 0x08
REG_VAL_MAG_MODE_ST = 0x10
# define ICM-20948 MAG Register  end

MAG_DATA_LEN = 6


class ICM20948Sensor:
    _instance = None
    Gyro = [0, 0, 0]
    Accel = [0, 0, 0]
    Mag = [0, 0, 0]
    pitch = 0.0
    roll = 0.0
    yaw = 0.0
    pu8data = [0, 0, 0, 0, 0, 0, 0, 0]
    U8tempX = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    U8tempY = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    U8tempZ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    GyroOffset = [0, 0, 0]
    Ki = 1.0
    Kp = 4.50
    q0 = 1.0
    q1 = q2 = q3 = 0.0
    angles = [0.0, 0.0, 0.0]
    MotionVal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, address: int = I2C_ADD_ICM20948) -> None:
        if ICM20948Sensor._instance is not None:
            raise RuntimeError("You cannot create another ICM20948Sensor class")

        if importlib.util.find_spec("smbus") is None:
            raise RuntimeError("smbus is required for ICM20948Sensor")

        smbus = importlib.import_module("smbus")
        self._address = address
        self._bus = smbus.SMBus(1)
        self.icm20948Check()  # Initialization of the device multiple times after power on will error
        time.sleep(0.5)  # We can skip this detection by delaying it by 500 milliseconds
        # user bank 0 register
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)
        self._write_byte(REG_ADD_PWR_MIGMT_1, REG_VAL_ALL_RGE_RESET)
        time.sleep(0.1)
        self._write_byte(REG_ADD_PWR_MIGMT_1, REG_VAL_RUN_MODE)
        # user bank 2 register
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_2)
        self._write_byte(REG_ADD_GYRO_SMPLRT_DIV, 0x07)
        self._write_byte(
            REG_ADD_GYRO_CONFIG_1,
            REG_VAL_BIT_GYRO_DLPCFG_6 | REG_VAL_BIT_GYRO_FS_1000DPS | REG_VAL_BIT_GYRO_DLPF,
        )
        self._write_byte(REG_ADD_ACCEL_SMPLRT_DIV_2, 0x07)
        self._write_byte(
            REG_ADD_ACCEL_CONFIG,
            REG_VAL_BIT_ACCEL_DLPCFG_6 | REG_VAL_BIT_ACCEL_FS_2g | REG_VAL_BIT_ACCEL_DLPF,
        )
        # user bank 0 register
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)
        time.sleep(0.1)
        self.icm20948GyroOffset()
        self.icm20948MagCheck()
        self.icm20948WriteSecondary(
            I2C_ADD_ICM20948_AK09916 | I2C_ADD_ICM20948_AK09916_WRITE,
            REG_ADD_MAG_CNTL2,
            REG_VAL_MAG_MODE_20HZ,
        )
        ICM20948Sensor._instance = self

    @classmethod
    def get_instance(cls) -> "ICM20948Sensor":
        if not cls._instance:
            cls._instance = ICM20948Sensor()
        return cls._instance

    def icm20948_Gyro_Accel_Read(self) -> None:
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)
        data = self._read_block(REG_ADD_ACCEL_XOUT_H, 12)
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_2)
        self.Accel[0] = (data[0] << 8) | data[1]
        self.Accel[1] = (data[2] << 8) | data[3]
        self.Accel[2] = (data[4] << 8) | data[5]
        self.Gyro[0] = ((data[6] << 8) | data[7]) - self.GyroOffset[0]
        self.Gyro[1] = ((data[8] << 8) | data[9]) - self.GyroOffset[1]
        self.Gyro[2] = ((data[10] << 8) | data[11]) - self.GyroOffset[2]
        if self.Accel[0] >= 32767:  # Solve the problem that Python shift will not overflow
            self.Accel[0] = self.Accel[0] - 65535
        elif self.Accel[0] <= -32767:
            self.Accel[0] = self.Accel[0] + 65535
        if self.Accel[1] >= 32767:
            self.Accel[1] = self.Accel[1] - 65535
        elif self.Accel[1] <= -32767:
            self.Accel[1] = self.Accel[1] + 65535
        if self.Accel[2] >= 32767:
            self.Accel[2] = self.Accel[2] - 65535
        elif self.Accel[2] <= -32767:
            self.Accel[2] = self.Accel[2] + 65535
        if self.Gyro[0] >= 32767:
            self.Gyro[0] = self.Gyro[0] - 65535
        elif self.Gyro[0] <= -32767:
            self.Gyro[0] = self.Gyro[0] + 65535
        if self.Gyro[1] >= 32767:
            self.Gyro[1] = self.Gyro[1] - 65535
        elif self.Gyro[1] <= -32767:
            self.Gyro[1] = self.Gyro[1] + 65535
        if self.Gyro[2] >= 32767:
            self.Gyro[2] = self.Gyro[2] - 65535
        elif self.Gyro[2] <= -32767:
            self.Gyro[2] = self.Gyro[2] + 65535

    def icm20948MagRead(self) -> None:
        counter = 20
        while counter > 0:
            time.sleep(0.01)
            self.icm20948ReadSecondary(
                I2C_ADD_ICM20948_AK09916 | I2C_ADD_ICM20948_AK09916_READ,
                REG_ADD_MAG_ST2,
                1,
            )
            if (self.pu8data[0] & 0x01) != 0:
                break
            counter -= 1
        if counter != 0:
            for i in range(0, 8):
                self.icm20948ReadSecondary(
                    I2C_ADD_ICM20948_AK09916 | I2C_ADD_ICM20948_AK09916_READ,
                    REG_ADD_MAG_DATA,
                    MAG_DATA_LEN,
                )
                self.U8tempX[i] = (self.pu8data[1] << 8) | self.pu8data[0]
                self.U8tempY[i] = (self.pu8data[3] << 8) | self.pu8data[2]
                self.U8tempZ[i] = (self.pu8data[5] << 8) | self.pu8data[4]
            self.Mag[0] = (
                self.U8tempX[0]
                + self.U8tempX[1]
                + self.U8tempX[2]
                + self.U8tempX[3]
                + self.U8tempX[4]
                + self.U8tempX[5]
                + self.U8tempX[6]
                + self.U8tempX[7]
            ) / 8
            self.Mag[1] = -(
                self.U8tempY[0]
                + self.U8tempY[1]
                + self.U8tempY[2]
                + self.U8tempY[3]
                + self.U8tempY[4]
                + self.U8tempY[5]
                + self.U8tempY[6]
                + self.U8tempY[7]
            ) / 8
            self.Mag[2] = -(
                self.U8tempZ[0]
                + self.U8tempZ[1]
                + self.U8tempZ[2]
                + self.U8tempZ[3]
                + self.U8tempZ[4]
                + self.U8tempZ[5]
                + self.U8tempZ[6]
                + self.U8tempZ[7]
            ) / 8
        if self.Mag[0] >= 32767:  # Solve the problem that Python shift will not overflow
            self.Mag[0] = self.Mag[0] - 65535
        elif self.Mag[0] <= -32767:
            self.Mag[0] = self.Mag[0] + 65535
        if self.Mag[1] >= 32767:
            self.Mag[1] = self.Mag[1] - 65535
        elif self.Mag[1] <= -32767:
            self.Mag[1] = self.Mag[1] + 65535
        if self.Mag[2] >= 32767:
            self.Mag[2] = self.Mag[2] - 65535
        elif self.Mag[2] <= -32767:
            self.Mag[2] = self.Mag[2] + 65535

    def icm20948ReadSecondary(self, u8I2CAddr: int, u8RegAddr: int, u8Len: int) -> None:
        self.u8Temp = 0
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3
        self._write_byte(REG_ADD_I2C_SLV0_ADDR, u8I2CAddr)
        self._write_byte(REG_ADD_I2C_SLV0_REG, u8RegAddr)
        self._write_byte(REG_ADD_I2C_SLV0_CTRL, REG_VAL_BIT_SLV0_EN | u8Len)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

        self.u8Temp = self._read_byte(REG_ADD_USER_CTRL)
        self.u8Temp |= REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, self.u8Temp)
        time.sleep(0.01)
        self.u8Temp &= ~REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, self.u8Temp)

        for i in range(0, u8Len):
            self.pu8data[i] = self._read_byte(REG_ADD_EXT_SENS_DATA_00 + i)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3

        self.u8Temp = self._read_byte(REG_ADD_I2C_SLV0_CTRL)
        self.u8Temp &= ~((REG_VAL_BIT_I2C_MST_EN) & (REG_VAL_BIT_MASK_LEN))
        self._write_byte(REG_ADD_I2C_SLV0_CTRL, self.u8Temp)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

    def icm20948WriteSecondary(self, u8I2CAddr: int, u8RegAddr: int, u8data: int) -> None:
        self.u8Temp = 0
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3
        self._write_byte(REG_ADD_I2C_SLV1_ADDR, u8I2CAddr)
        self._write_byte(REG_ADD_I2C_SLV1_REG, u8RegAddr)
        self._write_byte(REG_ADD_I2C_SLV1_DO, u8data)
        self._write_byte(REG_ADD_I2C_SLV1_CTRL, REG_VAL_BIT_SLV0_EN | 1)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

        self.u8Temp = self._read_byte(REG_ADD_USER_CTRL)
        self.u8Temp |= REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, self.u8Temp)
        time.sleep(0.01)
        self.u8Temp &= ~REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, self.u8Temp)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3

        self.u8Temp = self._read_byte(REG_ADD_I2C_SLV0_CTRL)
        self.u8Temp &= ~((REG_VAL_BIT_I2C_MST_EN) & (REG_VAL_BIT_MASK_LEN))
        self._write_byte(REG_ADD_I2C_SLV0_CTRL, self.u8Temp)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

    def icm20948GyroOffset(self) -> None:
        s32TempGx = 0
        s32TempGy = 0
        s32TempGz = 0
        for _ in range(0, 32):
            self.icm20948_Gyro_Accel_Read()
            s32TempGx += self.Gyro[0]
            s32TempGy += self.Gyro[1]
            s32TempGz += self.Gyro[2]
            time.sleep(0.01)
        self.GyroOffset[0] = s32TempGx >> 5
        self.GyroOffset[1] = s32TempGy >> 5
        self.GyroOffset[2] = s32TempGz >> 5

    def _read_byte(self, cmd: int) -> int:
        return self._bus.read_byte_data(self._address, cmd)

    def _read_block(self, reg: int, length: int = 1) -> list[int]:
        return self._bus.read_i2c_block_data(self._address, reg, length)

    def _read_u16(self, cmd: int) -> int:
        lsb = self._bus.read_byte_data(self._address, cmd)
        msb = self._bus.read_byte_data(self._address, cmd + 1)
        return (msb << 8) + lsb

    def _write_byte(self, cmd: int, val: int) -> None:
        self._bus.write_byte_data(self._address, cmd, val)
        time.sleep(0.0001)

    def imuAHRSupdate(self, gx: float, gy: float, gz: float, ax: float, ay: float, az: float, mx: float, my: float, mz: float) -> None:
        norm = 0.0
        hx = hy = hz = bx = bz = 0.0
        vx = vy = vz = wx = wy = wz = 0.0
        exInt = eyInt = ezInt = 0.0
        ex = ey = ez = 0.0
        halfT = 0.024
        q0q0 = self.q0 * self.q0
        q0q1 = self.q0 * self.q1
        q0q2 = self.q0 * self.q2
        q0q3 = self.q0 * self.q3
        q1q1 = self.q1 * self.q1
        q1q2 = self.q1 * self.q2
        q1q3 = self.q1 * self.q3
        q2q2 = self.q2 * self.q2
        q2q3 = self.q2 * self.q3
        q3q3 = self.q3 * self.q3

        norm = float(1 / math.sqrt(ax * ax + ay * ay + az * az))
        ax = ax * norm
        ay = ay * norm
        az = az * norm

        norm = float(1 / math.sqrt(mx * mx + my * my + mz * mz))
        mx = mx * norm
        my = my * norm
        mz = mz * norm

        # compute reference direction of flux
        hx = 2 * mx * (0.5 - q2q2 - q3q3) + 2 * my * (q1q2 - q0q3) + 2 * mz * (q1q3 + q0q2)
        hy = 2 * mx * (q1q2 + q0q3) + 2 * my * (0.5 - q1q1 - q3q3) + 2 * mz * (q2q3 - q0q1)
        hz = 2 * mx * (q1q3 - q0q2) + 2 * my * (q2q3 + q0q1) + 2 * mz * (0.5 - q1q1 - q2q2)
        bx = math.sqrt((hx * hx) + (hy * hy))
        bz = hz

        # estimated direction of gravity and flux (v and w)
        vx = 2 * (q1q3 - q0q2)
        vy = 2 * (q0q1 + q2q3)
        vz = q0q0 - q1q1 - q2q2 + q3q3
        wx = 2 * bx * (0.5 - q2q2 - q3q3) + 2 * bz * (q1q3 - q0q2)
        wy = 2 * bx * (q1q2 - q0q3) + 2 * bz * (q0q1 + q2q3)
        wz = 2 * bx * (q0q2 + q1q3) + 2 * bz * (0.5 - q1q1 - q2q2)

        # error is sum of cross product between reference direction of fields and direction measured by sensors
        ex = (ay * vz - az * vy) + (my * wz - mz * wy)
        ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
        ez = (ax * vy - ay * vx) + (mx * wy - my * wx)

        if (ex != 0.0 and ey != 0.0 and ez != 0.0):
            exInt = exInt + ex * self.Ki * halfT
            eyInt = eyInt + ey * self.Ki * halfT
            ezInt = ezInt + ez * self.Ki * halfT

            gx = gx + self.Kp * ex + exInt
            gy = gy + self.Kp * ey + eyInt
            gz = gz + self.Kp * ez + ezInt

        self.q0 = self.q0 + (-self.q1 * gx - self.q2 * gy - self.q3 * gz) * halfT
        self.q1 = self.q1 + (self.q0 * gx + self.q2 * gz - self.q3 * gy) * halfT
        self.q2 = self.q2 + (self.q0 * gy - self.q1 * gz + self.q3 * gx) * halfT
        self.q3 = self.q3 + (self.q0 * gz + self.q1 * gy - self.q2 * gx) * halfT

        norm = float(1 / math.sqrt(self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3))
        self.q0 = self.q0 * norm
        self.q1 = self.q1 * norm
        self.q2 = self.q2 * norm
        self.q3 = self.q3 * norm

    def icm20948Check(self) -> bool:
        bRet = False
        if REG_VAL_WIA == self._read_byte(REG_ADD_WIA):
            bRet = True
        return bRet

    def icm20948MagCheck(self) -> bool:
        self.icm20948ReadSecondary(
            I2C_ADD_ICM20948_AK09916 | I2C_ADD_ICM20948_AK09916_READ,
            REG_ADD_MAG_WIA1,
            2,
        )
        if (self.pu8data[0] == REG_VAL_MAG_WIA1) and (self.pu8data[1] == REG_VAL_MAG_WIA2):
            bRet = True
            return bRet
        return False

    def icm20948CalAvgValue(self) -> None:
        self.MotionVal[0] = self.Gyro[0] / 32.8
        self.MotionVal[1] = self.Gyro[1] / 32.8
        self.MotionVal[2] = self.Gyro[2] / 32.8
        self.MotionVal[3] = self.Accel[0]
        self.MotionVal[4] = self.Accel[1]
        self.MotionVal[5] = self.Accel[2]
        self.MotionVal[6] = self.Mag[0]
        self.MotionVal[7] = self.Mag[1]
        self.MotionVal[8] = self.Mag[2]

    def calc_pitch_degrees(self) -> float:
        self.pitch = math.asin(-2 * self.q1 * self.q3 + 2 * self.q0 * self.q2) * 57.3
        return self.pitch

    def calc_roll_degrees(self) -> float:
        self.roll = (
            math.atan2(2 * self.q2 * self.q3 + 2 * self.q0 * self.q1, -2 * self.q1 * self.q1 - 2 * self.q2 * self.q2 + 1)
            * 57.3
        ) + 90.0
        return self.roll

    def calc_yaw_degrees(self) -> float:
        self.yaw = (
            math.atan2(-2 * self.q1 * self.q2 - 2 * self.q0 * self.q3, 2 * self.q2 * self.q2 + 2 * self.q3 * self.q3 - 1)
            * 57.3
        )
        return self.yaw


def _demo() -> None:
    print("\nSense HAT Test Program ...\n")
    icm20948 = ICM20948Sensor()
    while True:
        icm20948.icm20948_Gyro_Accel_Read()
        icm20948.icm20948MagRead()
        icm20948.icm20948CalAvgValue()
        time.sleep(0.1)
        icm20948.imuAHRSupdate(
            icm20948.MotionVal[0] * 0.0175,
            icm20948.MotionVal[1] * 0.0175,
            icm20948.MotionVal[2] * 0.0175,
            icm20948.MotionVal[3],
            icm20948.MotionVal[4],
            icm20948.MotionVal[5],
            icm20948.MotionVal[6],
            icm20948.MotionVal[7],
            icm20948.MotionVal[8],
        )
        icm20948.calc_pitch_degrees()
        icm20948.calc_roll_degrees()
        icm20948.calc_yaw_degrees()
        print("\r\n /-------------------------------------------------------------/ \r\n")
        print("\r\n Roll = %.2f , Pitch = %.2f , Yaw = %.2f\r\n" % (icm20948.roll, icm20948.pitch, icm20948.yaw))
        print("\r\nAcceleration:  X = %d , Y = %d , Z = %d\r\n" % (icm20948.Accel[0], icm20948.Accel[1], icm20948.Accel[2]))
        print("\r\nGyroscope:     X = %d , Y = %d , Z = %d\r\n" % (icm20948.Gyro[0], icm20948.Gyro[1], icm20948.Gyro[2]))
        print("\r\nMagnetic:      X = %d , Y = %d , Z = %d" % ((icm20948.Mag[0]), icm20948.Mag[1], icm20948.Mag[2]))


if __name__ == "__main__":
    _demo()
