import skinematics as skin
import numpy as np
import quaternion
import pandas as pd
# from madgwick_py import madgwickahrs
from CMadgwick import CMad
from scipy.spatial.transform import Rotation as R


def orientation(gyro, accl, cmad):

    gx, gy, gz = gyro
    ax, ay, az = accl

    quat = cmad.MadgwickAHRSupdateIMU(gx, gy, gz, ax, ay, az)
    # quat = np.quaternion(quat[0], quat[1], quat[2], quat[3])
    # print(quat)
    return quat


if __name__=='__main__':
    path = '../100GOPRO/kandidaten/csv/'
    file = '2_5-'

    gyroAcclGps_pd = pd.read_csv(path + file + 'gyroAcclGpsRA.csv')
    gyroAcclGps = gyroAcclGps_pd.to_numpy()

    gyro = gyroAcclGps[:, 3:6]
    accl = gyroAcclGps[:, 6:9]

    rpy = np.zeros((gyro.shape[0], 3))
    wxyz = np.zeros((gyro.shape[0], 4))

    cmad = CMad()
    for i in range(len(gyro)):
        if not np.isnan(gyroAcclGps[i][4]):
            quat = orientation(gyro[i], accl[i], cmad)
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            wxyz[i] = quat
            xyz = r.as_euler('xyz', degrees=False)
            rpy[i] = xyz

    gyroAcclGpsABGWXYZ = gyroAcclGps_pd.assign(X=rpy[:,0], Y=rpy[:,1], Z=rpy[:,2], Qw=wxyz[:,0], Qx=wxyz[:,1], Qy=wxyz[:,2], Qz=wxyz[:,3])
    # gyroAcclGpsABGWXYZ.to_csv(path + file + 'xyzQuat.csv')
