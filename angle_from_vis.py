import numpy as np
import pandas as pd
from birdview import birdview, get_homography2
from camera_K import K_homog as K
from mark_lanes import mark_lanes
from cordFrames import cordFrame, worldFrame, transform, get_cordFrames
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from scipy.stats import linregress
from skimage.draw import circle_perimeter, line_aa, bezier_curve
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick import CMad
import skimage.transform as tf


class IIR_filter():
    def __init__(self, value=1, alpha=0.5):
        self.value = value
        self.alpha = alpha
    def update(self, new_value):
        self.value = self.alpha * new_value + (1-self.alpha) * self.value
        return self.value

'''
Takes frames from a video file and roll-angle from gyro-data to calculate the pitch-angle of the camera
relative to the road, saves in csv and birdview as PNGs
'''


def angle_from_vis():
    angle_calc = None
    world_frame, fw_frame, cam_frame, imu_frame, mc_frame = get_cordFrames()
    csv_path = '../100GOPRO/testfahrt_1006/kandidaten/csv/'
    video_path = '../100GOPRO/testfahrt_1006/kandidaten/'
    file = '3_2'
    start_msec = 0
    font = cv.FONT_HERSHEY_SIMPLEX
    grav_center = 2.805
    calib_pitch_from_vis = 0.698131700797732

    angle_filter = IIR_filter(calib_pitch_from_vis, alpha=0.3)
    px_m_filter = IIR_filter(180, alpha=0.5)

    # get data from csv as array, ignore first two elements to resolve empty datapoints
    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwickQuat.csv')[['Milliseconds','Radius','Quat']].to_numpy()[2:].swapaxes(1,0)
    cap = cv.VideoCapture(video_path + file + '.mp4')
    pitch_angles = np.zeros((np.int32(cap.get(7)),2))
    cap.set(0, start_msec)

    # int counting frames(for enumerating images)
    frame_nr_int = 0
    while(cap.isOpened()):
        # capture frame-by-frame
        ret, frame = cap.read()
        # print(ret)
        # frame = cv.imread('./problematic0.png')
        # frame = cv.imread('./calib_yb.png')


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

            # marked_im, hough_im, retval = mark_lanes(frame, 0.340036930698958)
            # marked_im, hough_im, retval = mark_lanes(frame, 0.234190505239813) # problematic0
            # marked_im, hough_im, retval = mark_lanes(frame, 0)
            # marked_im, hough_im, retval = mark_lanes(frame, -0.298919415637517)

            bird_im = np.zeros((frame.shape[1], frame.shape[0], 3))

            # if coloring not worked, use same marking positions as in last frame
            if not retval:
                print('not retval')
                yellow = np.where(np.all(last_frame == [0,255,255], axis=-1))
                blue = np.where(np.all(last_frame == [255,0,0], axis=-1))
                frame[yellow] = [0,255,255]
                frame[blue] = [255,0,0]
                marked_im = frame

            # find angle via iterating over a transform
            # closing in on a birdview (==> road markings equidistant)
            bird_im, angle_calc, found, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin = birdview(marked_im, False, angle_calc)

            if found:
                print('found')
                print(angle_filter.value)
                angle_calc = angle_filter.update(angle_calc)
            else:
                angle_calc = angle_filter.value

            print(angle_filter.value)
            pitch_angles[frame_nr_int] = mil, angle_calc
            # print(np.degrees(angle_calc))
            # print((slope, inter1, inter2))

            # scale for visibiliy, calculated from pixel/meter information from calibration
            # should not be necessary (bug)
            scale = tf.AffineTransform(scale=(1, 1.739))
            scaled_img = tf.warp(bird_im, inverse_map=scale)

            # remove lower black rows
            nonblack = np.amax(np.where(np.all(scaled_img!=[0,0,0], axis=-1))[0])
            scaled_img = scaled_img[:nonblack]
            # cv.imwrite('rotated_and_scaled.png', scaled_img)
            bird_im = scaled_img

            # get cam_origin in center and reduce width
            # print(cam_origin)
            cam_origin = np.where(np.all(bird_im == [0,0,255], axis=-1))
            cam_origin = (cam_origin[0][0], cam_origin[1][0])

            # print(cam_origin)
            bird_im = bird_im[:, max(0, cam_origin[1]-1000):min(cam_origin[1]+1000, 4000)]
            cam_origin = cam_origin[0], 1000

            # get width of lane in px
            road_mark_distance = dist(slope, inter1, inter2)
            print('mark distance:')
            print(road_mark_distance)

            # get width of one meter in px (lane width == 3m)
            pixel_meter = road_mark_distance / 3
            print(pixel_meter)

            pixel_meter = px_m_filter.update(pixel_meter)

            # radius = -111.83459873843
            # radius = -130.946140709033 # problematic0
            # radius = -130946140709033

            # get radius and center of mass of motorcycle in px
            radius_pixel = radius * pixel_meter
            grav_center_px = grav_center * pixel_meter
            fw_point = cam_origin


            print(grav_center_px)
            print(radius_pixel)
            print('slope:')
            print(slope)

            # slope of line orthogonal to driving direction
            perp_slope = -1/slope
            # center of curve is radius from motorcycle
            circle_center = (fw_point[0], fw_point[1]-radius_pixel)
            print(circle_center)
            rr,cc = circle_perimeter(np.int64(circle_center[0]), np.int64(circle_center[1]), np.int64(np.abs(radius_pixel)), shape = bird_im.shape)

            # print(perimeter[0].shape)
            # print(perimeter)
            # print(rr,cc)

            # color curve
            bird_im[rr,cc] = [0,0,255]
            line_thickness = 10
            rr[rr<line_thickness] = 0
            rr[rr>bird_im.shape[0] - line_thickness] = 0

            for i in range(1,line_thickness):
                bird_im[rr+i, cc] = [0,0,255]
                bird_im[rr-i, cc] = [0,0,255]

            bird_im = cv.resize(bird_im, (np.int32(bird_im.shape[1]/2), np.int32(bird_im.shape[0]/2)))


            # cv.imwrite('finally.png', bird_im)

            print('saving...')
            print('../100GOPRO/testfahrt_1006/kandidaten/%s_birdview/%s.png'%(file, frame_nr_int))
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_birdview/%s.png'%(file, frame_nr_int), bird_im)
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_marked/%s.png'%(file, frame_nr_int), marked_im)
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_hough/%s.png'%(file, frame_nr_int), hough_im)
            # print(marked_im.shape)
            cv.imshow(file, marked_im)
            print(frame_nr_int)
            last_frame = marked_im
            frame_nr_int += 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            pitch_df = pd.DataFrame(data=pitch_angles, columns=['Milliseconds', 'Pitch_from_vis'])
            # print(csv_path + file + '-pitch.csv')
            # pitch_df.to_csv(csv_path + file + '-pitch.csv')

            # When everything done, release the capture
            cap.release()
            cv.destroyAllWindows()
            quit()


# return index of value in array closest to input vale
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return idx

# get distance of parallel lines with intersects and slope
def dist(m, b1, b2):
    d = np.abs(b2 - b1) / (np.sqrt(m ** 2 + 1));
    return d

'''
def dy(distance, m):
    return m*dx(distance, m)

def dx(distance, m):
    return distance * np.sqrt(1/((m**2)+1))
'''
if __name__=='__main__':
    angle_from_vis()
