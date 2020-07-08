import numpy as np
import pandas as pd
from birdview import birdview
from mark_lanes import mark_lanes
from cordFrames import cordFrame, worldFrame, transform, get_cordFrames
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from scipy.stats import linregress
from skimage.draw import circle_perimeter, line_aa, bezier_curve
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick import CMad

'''
Plays frames from a video file and calculates the pitch angle of the camera
relative to the road.
Takes Roll-Angle from Gyro and writes angle in each frame (if found) to csv.
'''

def angle_from_vis():
    angle_calc = None
    world_frame, fw_frame, cam_frame, imu_frame = get_cordFrames()
    csv_path = '../100GOPRO/testfahrt_1006/kandidaten/csv/'
    video_path = '../100GOPRO/testfahrt_1006/kandidaten/'
    file = '3_2'
    start_msec = 4111.0
    font = cv.FONT_HERSHEY_SIMPLEX


    # get data from csv as array, ignore first two elements to resolve empty datapoints
    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwickQuat.csv')[['Milliseconds','Radius','Quat']].to_numpy()[2:].swapaxes(1,0)
    # print(radius_madgwick)

    # radius_madgwick[0] -= radius_madgwick[0,0]


    cap = cv.VideoCapture(video_path + file + '.mp4')
    pitch_angles = np.zeros((np.int32(cap.get(7)),2))
    cap.set(0, start_msec)
    # int counting frames(for saving snapshots)
    frame_nr_int = 0
    while(cap.isOpened()):
        # capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            height,width = frame.shape[:2]
            # find closest Datapoint in time to current frame
            nearest_rad = find_nearest(radius_madgwick[0], cap.get(0))
            mil, radius, ori = radius_madgwick[:,nearest_rad]

            # cast Quat from string(pandas) to float
            if type(ori)==str:
                ori = [np.float64(x.strip(' []')) for x in ori.split(',')]

            # rearrange to scalar-last format, cast to Euler
            ori = [ori[1],ori[2],ori[3],ori[0]]
            ori_eul = R.from_quat(ori).as_euler('xyz',degrees=False)

            # first step: color 2 lines belonging to road markings in video
            marked_im, hough_im, retval = mark_lanes(frame, -ori_eul[2])
            bird_im = np.zeros((frame.shape[1], frame.shape[0], 3))

            # if coloring worked, find angle via iterating over a transform
            # closing in on a birdview (==> road markings parallel)
            if retval:
                bird_im, angle_calc, found = birdview(marked_im, False, angle_calc)
                if found:
                    # print(angle_calc)
                    # quit()
                    pitch_angles[frame_nr_int] = mil, angle_calc
                    print(np.degrees(angle_calc))

                    yellow_warp = np.where(np.all(bird_im == [0,255,255], axis=-1))
                    blue_warp = np.where(np.all(bird_im == [255,0,0], axis=-1))
                    if blue_warp[0].size > 0 and yellow_warp[0].size > 0:

                        slope_y, intercept_y = linregress(yellow_warp[1], yellow_warp[0])[:2]
                        slope_b, intercept_b = linregress(blue_warp[1], blue_warp[0])[:2]
                        print((slope_y, slope_b))
                        road_mark_distance = dist(slope_y, intercept_y, intercept_b)
                        print('mark distance:')
                        print(road_mark_distance)
                        pixel_meter = road_mark_distance / 3
                        fw_point = (5000, 2000)
                        radius_pixel = radius * pixel_meter
                        print(radius_pixel)
                        perimeter = circle_perimeter(fw_point[0], np.int64(fw_point[1] - radius_pixel), np.int64(radius_pixel), shape = bird_im.shape)

                        bird_im[perimeter] = [0,255,0]



                else:
                    print('No fitting angle found. Best guess:')
                    print(angle_calc)
            else:
                print('no lines found in image')
                pitch_angles[frame_nr_int] = mil, None

            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_birdview_2/%s.png'%(file, frame_nr_int), bird_im)
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_marked_2/%s.png'%(file, frame_nr_int), marked_im)
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_hough_2/%s.png'%(file, frame_nr_int), hough_im)
            # print(marked_im.shape)
            cv.imshow(file, marked_im)
            print(frame_nr_int)
            frame_nr_int += 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            pitch_df = pd.DataFrame(data=pitch_angles, columns=['Milliseconds', 'Pitch_from_vis'])
            # print(csv_path + file + '-pitch.csv')
            pitch_df.to_csv(csv_path + file + '-pitch.csv')

            # When everything done, release the capture
            cap.release()
            cv.destroyAllWindows()
            quit()


# return index of value closest to input vale
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return idx

def dist(m, b1, b2):
    d = np.abs(b2 - b1) / (np.sqrt(m ** 2) + 1);
    return d

if __name__=='__main__':
    angle_from_vis()
