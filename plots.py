import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation  as R
import matplotlib
'''
Visualize data in plots.

'''
# complete_file = '../100GOPRO/testfahrt_1006/kandidaten/csv/3_0-gyroAcclGpsMadgwickQuat.csv'
# gps_file = '../100GOPRO/kandidaten/csv/3_2-gps.csv'
# gyro_file = '../100GOPRO/kandidaten/csv/2_3-gyro.csv'
# accl_file = '../100GOPRO/kandidaten/csv/3_2-accl.csv'
def plot_gps(filename):

    gps_complete = pd.read_csv(filename)

    gps_latlong = gps_complete[['Latitude', 'Longitude']]

    print(gps_latlong.head())

    plt.figure(figsize=(20,10))
    t = gps_complete['Milliseconds']
    lat = gps_latlong['Latitude']
    long = gps_latlong['Longitude']
    plt.plot(long, lat)
    plt.title('GPS')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    for i, ms in enumerate(t):
        if i % 20==0:
            x = long[i]
            y = lat[i]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, ms, fontsize=9, color='red')
    plt.scatter(x, y, marker='x', color='red', label='time in ms')
    leg = plt.legend()
    plt.savefig(filename + '.png')
    plt.show()

def plot_angles(filename):
    complete = pd.read_csv(filename)

    pitch_from_vis = pd.read_csv('../100GOPRO/testfahrt_1006/kandidaten/csv/3_2-pitch.csv')
    pitch_time, pitch = pitch_from_vis[['Milliseconds', 'Pitch_from_vis']].to_numpy()[2:].swapaxes(1,0)
    pitch_time = pitch_time.astype(np.float64)
    pitch = pitch.astype(np.float64)

    print(pitch.shape)
    print(pitch_time.shape)
    # pitch = pitch[pitch!=0]
    # pitch_time = pitch_time[pitch_time!=0]
    # pitch = np.insert(pitch, 0, 0)
    pitch = np.degrees(pitch)
    # print(pitch)
    # print(pitch_time)
    # quit()

    time, vel, quat = complete[['Milliseconds','Speed','Quat']].to_numpy()[2:].swapaxes(1,0)
    vel = vel.astype(np.float64)
    time = time.astype(np.float64)
    veltime= np.stack([vel,time], axis=1)
    # print(veltime)
    # print(veltime.shape)

    veltime = veltime[~np.isnan(veltime).any(axis=1)]
    vel_diff = np.diff(veltime[:,0], append=veltime[-1, 0])
    # print(veltime.shape)
    # print(veltime)
    # quit()
    # print(time)
    # print(quat.dtype)
    # print(quat.shape)
    quat = [np.float64(x.strip(' []').split(',')) for x in quat]#.split(',')]
    # print(quat[0])
    quat = np.roll(quat, -1)
    # print(quat[0])
    eul = np.zeros((quat.shape[0], 3))
    for q in np.arange(quat.shape[0]):
        eul[q] = R.from_quat(quat[q]).as_euler('xyz', degrees=True)
    # print(eul)

    fig, ax1 = plt.subplots()
    plt.title('Angles over time')
    # m = np.mean(pitch[np.isnan(pitch)==False])
    # print(m)
    # quit()
    # ax1.plot(time, eul[:,0], label= 'X', linewidth=1, color='red')
    ax1.plot(time, eul[:,1], label= 'Y', linewidth=1, color= 'green')
    print(np.mean(eul[:1]))
    # ax1.plot(time, eul[:,2], label= 'Z', linewidth=1, color='blue')
    ax1.plot(pitch_time, np.full_like(pitch_time, fill_value=41), label= 'asd', linewidth=1, color='black')
    ax1.scatter(pitch_time[1:], pitch[1:], label= 'pitch from vis', linewidth=1, color='gold')
    plt.xlabel('time in ms')
    plt.ylabel('Angle in degree')

    # plt.ylim(35,45)
    leg = plt.legend()
    for lh in leg.legendHandles:
        if type(lh) is not matplotlib.collections.PathCollection:
            lh._legmarker.set_alpha(0)

    ax2 = ax1.twinx()
    #
    # vel = veltime[:,0] - np.mean(veltime[:,0])
    vel_t = veltime[:,1]
    #
    # ax2.plot(vel_t, vel, label= 'Velocity minus mean (%s)'%(np.mean(veltime[:,0])), linewidth=1, color='black')   #TODO radius Radius in radius.py
    ax2.plot(vel_t, vel_diff, label= 'Acceleration', linewidth=1, color='purple')   #TODO radius Radius in radius.py
    # # ax2.plot(vel_t, np.full_like(vel_t, fill_value=np.mean(vel)) , label= str(np.mean(vel)), linewidth=1, color='darkgreen')   #TODO radius Radius in radius.py
    # plt.ylim(np.amin(vel), np.amax(vel), 25)
    plt.ylabel('Vel in m/s')
    leg = plt.legend()#loc=9)
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(0)


    fig.tight_layout()
    plt.gcf().set_size_inches(20, 5)
    plt.savefig(filename  + 'pitch' + '.png')



def plot_gyro(filename):

    complete = pd.read_csv(filename)

    gyro_complete = complete[['Milliseconds', 'GyroX', 'GyroY', 'GyroZ']]
    RadAng = complete[['Milliseconds', 'Radius', 'Angle']]
    # euler = complete[['Milliseconds', 'X', 'Y', 'Z']]

    # print(gyro_complete.head())
    # print(RadAng.head())


    time = gyro_complete[['Milliseconds']].to_numpy(dtype='float64')
    print(type(time))
    # plt.figure(figsize=(100,10))
    fig, ax1 = plt.subplots()

    plt.title('Gyro-Data: Right turn')

    ax1.plot(time, gyro_complete[['GyroX']].to_numpy(dtype='float64'), label= 'GyroX', linewidth=1)
    ax1.plot(time, gyro_complete[['GyroY']].to_numpy(dtype='float64'), label= 'GyroY', linewidth=1)
    ax1.plot(time, gyro_complete[['GyroZ']].to_numpy(dtype='float64'), label= 'GyroZ', linewidth=1)
    ax1.plot(time, RadAng[['Angle']].to_numpy(dtype='float64'), label= 'MadAngle', linewidth=1)
    # ax1.plot(time, euler[['X']], label= 'EulerX', linewidth=0.2)
    # ax1.plot(time, euler[['Y']], label= 'EulerY', linewidth=0.2)
    # ax1.plot(time, euler[['Z']], label= 'MadAngle', linewidth=0.2)

    plt.xlabel('time in ms')
    plt.ylabel('value of sensor in rad/s\nAngle in rad')

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(0)

    ax2 = ax1.twinx()
    ax2.plot(time, RadAng[['Radius']], label= 'MadRadius', linewidth=1, color='black')   #TODO radius Radius in radius.py
    plt.ylim(-100, 100)
    plt.ylabel('Radius in m')
    leg = plt.legend()#loc=9)
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(0)
    fig.tight_layout()
    plt.gcf().set_size_inches(200, 10)
    plt.savefig(filename + '.png')
    # plt.show()

def plot_accl(filename):

        accl_complete = pd.read_csv(filename)

        print(accl_complete.head())
        plt.figure(figsize=(20,10))
        time = accl_complete[['Milliseconds']]

        plt.plot(time, accl_complete[['AcclX']], label= 'X', linewidth=0.1)
        plt.plot(time, accl_complete[['AcclY']], label= 'Y', linewidth=0.1)
        plt.plot(time, accl_complete[['AcclZ']], label= 'Z', linewidth=0.1)
        plt.title('Scatter plot accl')
        plt.xlabel('time in ms')
        plt.ylabel('value of sensor')
        leg = plt.legend()
        for lh in leg.legendHandles:
            lh._legmarker.set_alpha(0)
        plt.savefig(filename + '.png')
        plt.show()


if __name__ == '__main__':
    # plot_gps(gps_file)
   # plot_gyro(complete_file)
#    plot_accl(accl_file)
    plot_angles('../100GOPRO/testfahrt_1006/kandidaten/csv/%s-gyroAcclGpsMadgwickQuat.csv'%('3_2'))
    # for f in ['1_0', '2_0', '2_1', '2_2', '2_3', '2_4', '3_0', '3_1']:
    #     plot_angles('../100GOPRO/testfahrt_1006/kandidaten/csv/%s-gyroAcclGpsMadgwickQuat.csv'%(f))
    #     print(f)
