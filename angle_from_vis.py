import numpy as np
import pandas as pd
from birdview import birdview, get_homography2
from camera_K import K_homog as K
# from mark_lanes import mark_lanes
from road_markings import get_markings
from cordFrames import cordFrame, worldFrame, transform, get_cordFrames
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from scipy.stats import linregress
from skimage.draw import circle_perimeter, line_aa, bezier_curve
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick import CMad
import skimage.transform as tf


'''
Plays frames from a video file and calculates the pitch angle of the camera
relative to the road.
Takes Roll-Angle from Gyro and writes angle in each frame (if found) to csv.
'''

def angle_from_vis():
    angle_calc = None
    world_frame, fw_frame, cam_frame, imu_frame, mc_frame = get_cordFrames()
    csv_path = '../100GOPRO/testfahrt_1006/kandidaten/csv/'
    video_path = '../100GOPRO/testfahrt_1006/kandidaten/'
    file = '3_2'
    start_msec = 4888.22155489#2819.48615282
    font = cv.FONT_HERSHEY_SIMPLEX
    grav_center = 2.805

    # get data from csv as array, ignore first two elements to resolve empty datapoints
    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwickQuat.csv')[['Milliseconds','Radius','Quat']].to_numpy()[2:].swapaxes(1,0)
    # print(radius_madgwick)

    # radius_madgwick[0] -= radius_madgwick[0,0]


    cap = cv.VideoCapture(video_path + file + '.mp4')
    pitch_angles = np.zeros((np.int32(cap.get(7)),2))
    cap.set(0, start_msec)
    # int counting frames(for saving snapshots)
    frame_nr_int = 293#169
    while(cap.isOpened()):
        # capture frame-by-frame
        ret, frame = cap.read()



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
            marked_im, hough_im, retval = get_markings(frame, ori_eul[2])
            # marked_im, hough_im, retval = mark_lanes(frame, -ori_eul[2])
            # marked_im, hough_im, retval = mark_lanes(frame, 0.340036930698958)
            # marked_im, hough_im, retval = mark_lanes(frame, 0.234190505239813) # problematic0.png
            # marked_im, hough_im, retval = mark_lanes(frame, 0)





            # marked_im, hough_im, retval = mark_lanes(frame, -0.298919415637517)
            bird_im = np.zeros((frame.shape[1], frame.shape[0], 3))

            # if coloring worked, find angle via iterating over a transform
            # closing in on a birdview (==> road markings equidistant)
            if retval:
                bird_im, angle_calc, found, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin = birdview(marked_im, False, angle_calc)
                if found and np.isnan(slope)==False:
                    # print(angle_calc)
                    # quit()

                    pitch_angles[frame_nr_int] = mil, angle_calc
                    print(np.degrees(angle_calc))

                    # yellow_warp = np.where(np.all(bird_im == [0,255,255], axis=-1))
                    # blue_warp = np.where(np.all(bird_im == [255,0,0], axis=-1))
                    # if blue_warp[0].size > 0 and yellow_warp[0].size > 0:

                        # slope_y, intercept_y = linregress(yellow_warp[1], yellow_warp[0])[:2]
                        # slope_b, intercept_b = linregress(blue_warp[1], blue_warp[0])[:2]
                    print((slope, inter1, inter2))
                    scale = tf.AffineTransform(scale=(1, 1.739))
                    scaled_img = tf.warp(bird_im, inverse_map=scale)
                    # scaled_img *= 255
                    bird_im = scaled_img
                    cv.imwrite('rotated_and_scaled.png', bird_im)
                    cam_origin = np.where(np.all(bird_im == [0,0,255], axis=-1))
                    cam_origin = (cam_origin[0][0], cam_origin[1][0])

                    last_nonblack_row = np.amax(np.where(np.all(bird_im!=[0,0,0], axis=-1))[0])
                    # print(np.where(np.all(bird_im!=[0,0,0], axis=-1)))
                    print(last_nonblack_row)
                    # quit()
                    bird_im = bird_im[:last_nonblack_row]

                    bird_im = bird_im[:, max(cam_origin[1]-1000, 0):min(cam_origin[1]+1000, 4000)]
                    cv.imwrite('rot_sc_cut.png', bird_im)
                    cam_origin = np.where(np.all(bird_im == [0,0,255], axis=-1))
                    cam_origin = (cam_origin[0][0], cam_origin[1][0])

                    road_mark_distance = dist(slope, inter1, inter2)
                    print('mark distance:')
                    print(road_mark_distance)
                    pixel_meter = road_mark_distance / 3
                    # fw_point = cam_origin[1], cam_origin[0]
                    print(pixel_meter)


                    # radius = -111.83459873843
                    # radius = -130.946140709033 # problematic0
                    # radius = -130946140709033
                    # radius =


                    radius_pixel = radius * pixel_meter
                    grav_center_px = grav_center * pixel_meter
                    fw_point = cam_origin
                    # fw_point = cam_origin[0] + dx(grav_center_px, 0), cam_origin[1] + dy(grav_center_px, 0)


                    # fw_point = cam_origin[0] + dx(grav_center_px, slope), cam_origin[1] + dy(grav_center_px, slope)



                    print(grav_center_px)
                    print(radius_pixel)
                    print('slope:')
                    print(slope)
                    perp_slope = -1/slope
                    # perp_slope = slope
                    # bird_im = cv.line(bird_im, pt1=(np.int32(fw_point[1]), fw_point[0]), pt2=(0, np.int32(-slope*bird_im.shape[0] - fw_point[1])), color=(0,255,0))
                    # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
                    # print(perp_slope)
                    # circle_center = (fw_point[1] + dx(radius_pixel, slope), fw_point[0] + dy(radius_pixel, slope))
                    # other_possible_circle_center = (fw_point[0] - dx(radius_pixel, slope), fw_point[1] - dy(radius_pixel, slope)) # going the other way
                    # print(circle_center)
                    # print(other_possible_circle_center)
                    '''
                    circle_center = (fw_point[0] + dx(radius_pixel, perp_slope), fw_point[1] + dy(radius_pixel, perp_slope))
                    other_possible_circle_center = (fw_point[0] - dx(radius_pixel, perp_slope), fw_point[1] - dy(radius_pixel, perp_slope)) # going the other way

                    print(circle_center)
                    print(other_possible_circle_center)
                    # bird_im = cv.line(bird_im, pt1=(np.int32(fw_point[1]), fw_point[0]), pt2=(np.int32(circle_center[1]), np.int32(circle_center[0])), color=(0,255,0))
                    print(fw_point)
                    '''
                    # if ori_eul[2] > 0:
                    #     rr,cc = circle_perimeter(np.int64(circle_center[0]), np.int64(circle_center[1]), np.int64(np.abs(radius_pixel)), shape = bird_im.shape)
                    # else:
                    #     rr,cc = circle_perimeter(np.int64(other_possible_circle_center[0]), np.int64(other_possible_circle_center[1]), np.int64(np.abs(radius_pixel)), shape = bird_im.shape)
                    # rr,cc = circle_perimeter(np.int64(other_possible_circle_center[0]), np.int64(other_possible_circle_center[1]), np.int64(np.abs(radius_pixel)), shape = bird_im.shape)
                    circle_center = (fw_point[0], fw_point[1]-radius_pixel)
                    # circle_center = (fw_point[0], fw_point[1])
                    print(circle_center)
                    rr,cc = circle_perimeter(np.int64(circle_center[0]), np.int64(circle_center[1]), np.int64(np.abs(radius_pixel)), shape = bird_im.shape)
                    # rr,cc = circle_perimeter(np.int64(circle_center[1]), np.int64(circle_center[0]), np.int64(np.abs(20)), shape = bird_im.shape)



                    # rr,cc = circle_perimeter(np.int64(circle_center[0]), np.int64(circle_center[1]), np.int64(np.abs(radius_pixel)), shape = bird_im.shape)

                    # print(perimeter[0].shape)
                    # print(perimeter)
                    # print(rr,cc)

                    bird_im[rr,cc] = [0,0,255]
                    line_thickness = 10

                    rr[rr<line_thickness] = 0
                    # cc[cc<line_thickness] = 0
                    rr[rr>bird_im.shape[0] - line_thickness] = 0
                    # cc[cc>bird_im.shape[1] - line_thickness] = 0
                    #
                    for i in range(1,line_thickness):
                        bird_im[rr+i, cc] = [0,0,255]
                        bird_im[rr-i, cc] = [0,0,255]


                    # bird_im[rr+1, cc+1] = [0,0,255]
                    # bird_im[rr+2, cc+2] = [0,0,255]
                    # bird_im[rr-1, cc-1] = [0,0,255]
                    # bird_im[rr-2, cc-2] = [0,0,255]


                    cv.imwrite('finally.png', bird_im)
                    #
                    # back_rot = R.from_euler('xyz', [0, angle_calc, 0], degrees=False).as_matrix()
                    # back_rot = np.linalg.inv(back_rot)
                    # H = get_homography2(back_rot, K)
                    # backwarp = back_warp(bird_im, H, dst_height=1080-630, dst_width=1920)
                    # quit()
                    bird_im = cv.resize(bird_im, (np.int32(bird_im.shape[1]/2), np.int32(bird_im.shape[0]/2)))
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
            # quit()

        if cv.waitKey(1) & 0xFF == ord('q'):
            pitch_df = pd.DataFrame(data=pitch_angles, columns=['Milliseconds', 'Pitch_from_vis'])
            # print(csv_path + file + '-pitch.csv')
            # pitch_df.to_csv(csv_path + file + '-pitch.csv')

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
    d = np.abs(b2 - b1) / (np.sqrt(m ** 2 + 1));
    return d


def dy(distance, m):
    return m*dx(distance, m)

def dx(distance, m):
    return distance * np.sqrt(1/((m**2)+1))

if __name__=='__main__':
    angle_from_vis()
