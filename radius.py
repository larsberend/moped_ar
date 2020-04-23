
import numpy as np
import pandas as pd
from CMadgwick import CMad
from scipy.spatial.transform import Rotation as R
import sys

'''
Calculates radius of a turn.
Input: gyro data in rad/s
gravitational accelaration
GPS-data: 2D velocity in m/s
timestamps of both
following formula described here:
https://de.wikipedia.org/wiki/Fahrphysik_(Fahrrad)
'''

class turn():
    def __init__(self, radius=1000, ang=0):
        self.radius = radius
        self.ang = ang

    def calcRad(self, vel, ang, g):
        self.radius = np.divide(np.power(vel, 2), np.multiply(np.tan(ang), g))
        return self.radius

    # Simple integration, no Madgwick-Filtering (deprecated)
    def calcAng(self, gyro0, gyro1):
        gyro0_Z = np.amin([gyro0[3], gyro1[3]])
        gyro1_Z = np.amax([gyro0[3], gyro1[3]])

        t = np.divide(np.subtract(gyro1[0], gyro0[0]), 1000)
        a = np.multiply(gyro1_Z, t)
        b = np.absolute(np.subtract(gyro1_Z, gyro0_Z))
        c = np.multiply(b, t)
        d = np.divide(c, 2)
        single_ang = np.subtract(a, d)

        self.ang = np.add(single_ang, self.ang)

        return self.ang

    def getRadius(self):
        return self.radius

# reads candidate-number from stdin, gets csv-file and extends it with radii,
# angles and quaternions(from madgwick filter)
if __name__=='__main__':
    path = '/mnt/c/Users/bb/Documents/Moped_AR/100GOPRO/kandidaten/csv/'
    file = sys.argv[1] + '-'
    test = turn()
    cmad = CMad()
    vel = np.nan
    gyroAcclGps_pd = pd.read_csv(path + file + 'gyroAcclGps.csv')#.iloc[1686:2749]
    # print(gyroAcclGps_pd.head())
    gyro = gyroAcclGps_pd[['GyroX', 'GyroY', 'GyroZ']].to_numpy()
    accl = gyroAcclGps_pd[['AcclX', 'AcclY', 'AcclZ']].to_numpy()
    gps = gyroAcclGps_pd[['Latitude', 'Longitude', 'Speed']].to_numpy()
    rows, cols = gyroAcclGps_pd.shape
    gyroAcclGps = np.full(shape=(rows, cols+2), fill_value=np.nan)
    gyroAcclGps[:, :-2] = gyroAcclGps_pd.to_numpy()
    quats = []
    print(gyro.shape)
    # print(gyroAcclGps.shape)
    # print(gyroAcclGps_pd.shape)
    # print(range(gyroAcclGps.shape[0]-1))
    for i in range(gyro.shape[0]):
        # check if datapoint from gyrometer, since not aligned
        if not np.isnan(gps[i][0]):
            vel = gps[i][2]
        # [t, gyroX, gyroY, gyroZ]
        # check if datapoint from GPS, since not aligned
        if not np.isnan(gyro[i][0]):
            gx, gy, gz = gyro[i]
            ax, ay, az = accl[i]

        # calc rotation relative to original position of IMU
        quat = cmad.MadgwickAHRSupdateIMU(gx, gy, gz, ax, ay, az)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        xyz = r.as_euler('xyz', degrees=False)
        # sanitiy check: no problmes with conversion from quat to euler angles
        # assert np.abs(np.dot(R.from_euler('xyz', xyz).as_quat(), r.as_quat()) - 1) < 0.00001
        # z-axis is roll-angle (rotation around axis in driving direction)
        angle = xyz[2]
        gyroAcclGps[i, -2] = angle

        radius = test.calcRad(vel, angle, 9.81)
        gyroAcclGps[i, -1] = radius
        quats.append(list(quat))

    gyroAcclGpsRA = gyroAcclGps_pd.assign(Radius=gyroAcclGps[:,-1], Angle=gyroAcclGps[:,-2], Quat=quats)
    print(gyroAcclGpsRA.head())
    print(gyroAcclGpsRA.dtypes)
    # quit()
    gyroAcclGpsRA.to_csv(path + file + 'gyroAcclGpsMadgwickQuat.csv')
