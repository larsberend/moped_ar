import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import mpl_toolkits
#from mpl_toolkits.basemap import Basemap
complete_file = '../100GOPRO/kandidaten/csv/2_complete-gyroAcclGpsMadgwick.csv'
gps_file = '../100GOPRO/kandidaten/csv/3_2-gps.csv'
gyro_file = '../100GOPRO/kandidaten/csv/3_2-gyro.csv'
accl_file = '../100GOPRO/kandidaten/csv/3_2-accl.csv'
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

    ax1.plot(time, gyro_complete[['GyroX']].to_numpy(dtype='float64'), label= 'GyroX', linewidth=0.2)
    ax1.plot(time, gyro_complete[['GyroY']].to_numpy(dtype='float64'), label= 'GyroY', linewidth=0.2)
    ax1.plot(time, gyro_complete[['GyroZ']].to_numpy(dtype='float64'), label= 'GyroZ', linewidth=0.2)
    ax1.plot(time, RadAng[['Angle']].to_numpy(dtype='float64'), label= 'MadAngle', linewidth=0.2)
    # ax1.plot(time, euler[['X']], label= 'EulerX', linewidth=0.2)
    # ax1.plot(time, euler[['Y']], label= 'EulerY', linewidth=0.2)
    # ax1.plot(time, euler[['Z']], label= 'MadAngle', linewidth=0.2)

    plt.xlabel('time in ms')
    plt.ylabel('value of sensor in rad/s\nAngle in rad')

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(0)

    ax2 = ax1.twinx()
    ax2.plot(time, RadAng[['Radius']], label= 'MadRadius', linewidth=0.3, color='black')   #TODO radius Radius in radius.py
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
   plot_gyro(complete_file)
#    plot_accl(accl_file)
