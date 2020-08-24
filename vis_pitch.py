import numpy as np
import cv2 as cv
import pandas as pd
from road_markings import get_markings
from birdview_new import find_bv

# colors = [[0,0,255], [255,255,0]]
colors = [[0,255,255], [255,0,0]]

def vis_pitch(img=None, known_roll=None):
    if img is not None:
        marked_img, ret = img, True
        if ret:
        # frame, roll = img, known_roll
        # marked_img, hough_img, ret = get_markings(frame, roll, colors[0], colors[1])
        # if ret:
            # print(marked_img.shape)
            cut_to_road = marked_img[540:, :]
            cv.imwrite('07_cut_to_road.png', cut_to_road)
            bv = find_bv(cut_to_road, colors[0], colors[1])
            if bv is not None:
                return 0, pitch
        return 0, 0
    start_msec = 0
    file = '3_2'
    path = '../100GOPRO/testfahrt_1006/kandidaten/'
    csv = pd.read_csv(path + 'csv/' + file + '-gyroAcclGpsMadgwickQuat.csv')[['Milliseconds', 'Angle']].to_numpy().swapaxes(1,0)
    # print(csv)
    cap = cv.VideoCapture(path + file + '.mp4')
    pitches = np.zeros((np.int32(cap.get(7)), 2))
    cap.set(0, start_msec)
    frame_nr = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        ms = cap.get(0)
        idx_csv = find_nearest(csv, ms)
        csv_ms, roll = csv[:, idx_csv]

        marked_img = get_markings(frame, roll, colors[0], colors[1])
        cut_to_road = marked_img[:, 540]
        pitch, bv = find_bv(cut_to_road, colors[0], colors[1])
        pitches[frame_nr] = ms, pitch
    # quit()
    return pitches

# return index of value closest to input vale
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return idx


if __name__ == '__main__':

    # img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    img, roll_angle= cv.imread('./calib_yb_rot.png'), 0
    print(vis_pitch(img, roll_angle))
    # print(vis_pitch())
