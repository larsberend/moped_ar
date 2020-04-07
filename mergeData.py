import pandas as pd
import numpy as np

def fromToCSV(gyro, accl, gps, path, file):
    gyroAccl = gyro.merge(accl, how='outer', on='Milliseconds')
    # print(gyro['Milliseconds'][1].type())
    gps['Milliseconds'] = gps['Milliseconds'].astype(np.float64)
    gyroAcclGps = gyroAccl.merge(gps, how='outer', on='Milliseconds')
    gyroAcclGps = gyroAcclGps.sort_values(by='Milliseconds', axis=0, ignore_index=True)
    gyroAcclGps.to_csv(path + file + 'gyroAcclGps.csv')

    print(gyroAcclGps.head())

if __name__=='__main__':
    path = '../100GOPRO/kandidaten/csv/'
    file = '2_complete-'
    gyro = pd.read_csv(path + file + 'gyro.csv')
    accl = pd.read_csv(path + file + 'accl.csv')
    gps = pd.read_csv(path + file + 'gps.csv')

    fromToCSV(gyro, accl, gps, path, file)
