import numpy as np
import pandas as pd
from birdview import birdview, get_homography2, warp_img
from mark_lanes import mark_lanes
from cordFrames import cordFrame, worldFrame, transform, get_cordFrames
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from skimage.draw import circle_perimeter, line_aa, bezier_curve
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick import CMad
from angle_from_vis import find_nearest
import skimage


# font = cv.FONT_HERSHEY_SIMPLEX
# focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
# pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
# K = np.dot(pixel_mat, focal_mat)

class Pitch_Filter():
    def __init__(self, p_vis, p_mad):
        self.last_p_vis = p_vis
        self.last_p_mad = p_mad
        self.count_mad = 1
        self.pitch = 0
    def update(self, p_vis, p_mad):
        # todo: wie im ersten schritt?

        if p_vis == 0:
            # the visual angle does not always find an angle that satisfies paralell
            self.count_mad += 1
            self.last_p_mad += p_mad
        else:
            self.last_p_mad /= self.count_mad
            self.count_mad = 1

            vis_correct = p_vis - self.pitch - self.last_p_mad + p_mad
            new_pitch = self.pitch - self.last_p_mad + p_mad + vis_correct
            self.pitch = new_pitch
            self.last_p_vis = p_vis
            self.last_p_mad = p_mad

'''
Plays frames from a video file and draws trajectory of the motorcycle in each of them.
Updates camera-parameters with data from IMU.
'''

def main():
    bv = True
    angle_calc = None
    world_frame, fw_frame, cam_frame, imu_frame, mc_frame = get_cordFrames()

    csv_path = '../100GOPRO/testfahrt_1006/kandidaten/csv/'
    video_path = '../100GOPRO/testfahrt_1006/kandidaten/'
    file = '3_2'
    start_msec = 000.0
    font = cv.FONT_HERSHEY_SIMPLEX
    calib_pitch_from_vis = 0.698131700797732

    # get data from csv as array, ignore first two elements to resolve empty datapoints
    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwickQuat.csv')[['Milliseconds','Radius','Quat']].to_numpy()[2:].swapaxes(1,0)
    # print(radius_madgwick)

    # radius_madgwick[0] -= radius_madgwick[0,0]
    if bv:
        pitch_df = pd.read_csv(csv_path + file + '-pitch.csv')
        pitch_np = pitch_df[['Milliseconds', 'Pitch_from_vis']].to_numpy().swapaxes(1,0)
        for i in np.arange(pitch_np[1].size):
            if pitch_np[1,i]!=0:
                last_value = pitch_np[1,i]
            else:
                pitch_np[1,i] = last_value

    cap = cv.VideoCapture(video_path + file + '.mp4')
    # out = cv.VideoWriter('3_2_curve.avi', cv.VideoWriter_fourcc('M','J','P','G'), 59.94, (1920,1080))
    cap.set(0, start_msec)
    # int counting frames(for saving snapshots)
    frame_nr_int = 0
    ret, orig_frame = cap.read()
    # ret, orig_frame = True, cv.imread('../circle_wdegrees.png')
    # orig_frame = cv.resize(orig_frame, (1920, 1080))
    while(cap.isOpened()):
        # capture frame-by-frame
        ret, frame = cap.read()
        # frame = orig_frame.copy()
        if ret:
            # print('ret')
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
            # ori_xy_fw = R.from_euler('xyz', [ori_eul[0], ori_eul[1], 0])
            ori_x_fw = R.from_euler('xyz', [ori_eul[0], 0, 0])
            # fw_trans = transform([0,0,0], ori_xy_fw.as_quat())
            fw_trans = transform([0,0,0], ori_x_fw.as_quat())
            # fw_frame.update_ori(fw_trans)

            ori_z_imu = R.from_euler('xyz',[0, 0, -2 * ori_eul[2]], degrees=False)
            # ori_z_imu = R.from_euler('xyz',[0, 0, -90], degrees=True)
            new_mc = ori_z_imu.apply([0,0.5,0])
            # print(new_mc)
            # new_mc = [new_mc[0], 0, -0.7105]
            new_mc = [new_mc[0], 0, 0.7105]
            mc_frame.update_ori(transform(new_mc, [0,0,0,1]))
            print(mc_frame)
            vis_pitch = None
            if bv == True:
                vis_pitch = pitch_np[1, frame_nr_int]
                print(pitch_np)
                # quit()
                if np.isnan(vis_pitch) == False:
                    # print(pitch_np)
                    # print(pitch_np.shape)
                    # print(mil)
                    pitch = vis_pitch - calib_pitch_from_vis

                    # rot_img = skimage.transform.rotate(frame, ori_eul[2], clip=True, preserve_range=True).astype(np.uint8)
                    # cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_rotated/%s.png'%(file, frame_nr_int), rot_img)
                    # H = get_homography2(vis_pitch, )


                    # cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_rotated/%s.png'%(file, frame_nr_int), rot_img)
                    # print(vis_pitch)
                    '''
                    if frame_nr_int==0:
                        pitch_filter = Pitch_Filter(vis_pitch, ori_eul[1])
                    else:
                        pitch_filter.update(vis_pitch, ori_eul[1])
                    pitch = pitch_filter.pitch
                    '''
                    print(pitch)

                    ori_yz = R.from_euler('xyz', [0, -pitch, -2 * ori_eul[2]])
                    # ori_yz = R.from_euler('xyz', [0, -pitch, -np.radians(frame_nr_int)])
                    print('\n')
                    print(-ori_eul[2])
                    print('\n')
                    pos_from_angle = ori_yz.apply([0, 1.1237, 0])
                    # print('pos_from_angle')
                    # print(pos_from_angle)
                    imu_trans = transform(pos_from_angle, [
                                                            [0, 0, np.sin(np.pi/4),np.cos(np.pi/4)],
                                                            [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)],
                                                            ori_yz.as_quat()
                                                            ,R.from_euler('xyz', [0,15,0], degrees=True).as_quat()
                                                            ]
                                         )
                    # check if Euler conversion returns same Quaternion:
                    # assert np.abs(np.dot(R.from_euler('xyz', ori_eul).as_quat(), ori) - 1) < 0.00001

                    # update fw_frame with pitch & yaw



            else:
                print('\n\n\nElse!! \n\n\n')
                ori_z_imu = R.from_euler('xyz',[0, 0, -np.radians(frame_nr_int)], degrees=False)
                # ori_z_imu_pos = R.from_euler('xyz',[0, 0, -ori_eul[2]], degrees=False)



                # ckeck translation of camera with angle of 0 and 90
                # ori_z_imu = R.from_euler('xyz',[0, 0, -np.pi/2],degrees=False)
                # ori_z_imu = R.from_euler('xyz',[0, 0, 0],degrees=False)

                print(ori_z_imu.as_quat())
                print(ori_z_imu.as_euler('xyz', degrees=True))


                # pos_from_angle = ori_yz_imu.apply([0, 1.1237, 0])
                pos_from_angle = ori_z_imu.apply([0, 1.1237, 0])

                # print(pos_from_angle)

                # calc new transform of IMU to world
                imu_trans = transform(pos_from_angle, [
                                                        # [0, 0, np.sin(np.pi/4),np.cos(np.pi/4)],
                                                        # [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)],
                                                        # [ori_z_imu[1], 0, 0, np.cos(np.arcsin(ori_z_imu[1]))]
                                                        ori_z_imu.as_quat()
                                                        # ori_yz_imu.as_quat()
                                                        # [0,np.sin(np.pi/64),0,np.cos(np.pi/64)]
                                                       ]
                                     )
            imu_frame.update_ori(imu_trans)
            print(imu_frame)

            cv.putText(frame, 'Frame Nr: %s'%(cap.get(0)), (1650,20), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'Radius: %s'%(radius), (1650,40), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'MIlliseconds(IMU): %s'%(mil), (1650,60), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)

            # calc visible trajectory in frame and view from above
            # curve_proj, top_view = draw_curve(radius, cam_frame, fw_frame, vis_pitch)
            # curve_proj, top_view = draw_curve(radius, cam_frame, fw_frame, None)
            curve_proj, top_view = draw_curve(radius, cam_frame, mc_frame, None)
            hori_proj, _ = draw_curve(radius, cam_frame, mc_frame, None, True)
            curve_proj=curve_proj[curve_proj[:,0]>=0].astype(np.int32)
            amount_lines = curve_proj.shape[0]-1
            for j in range(curve_proj.shape[0]-1):
                cv.line(frame, (curve_proj[j,0], curve_proj[j,1]), (curve_proj[j+1,0], curve_proj[j+1,1]), (0,0,255), thickness = 5)

            hori_proj=hori_proj[hori_proj[:,0]>=0].astype(np.int32)
            amount_l = hori_proj.shape[0]-1
            for j in range(hori_proj.shape[0]-1):
                cv.line(frame, (hori_proj[j,0], hori_proj[j,1]), (hori_proj[j+1,0], hori_proj[j+1,1]), (0,255,0), thickness = 3)


            x_offset = width - top_view.shape[1]
            y_offset = np.int(height/3) - top_view.shape[0]
            frame[y_offset:y_offset+top_view.shape[0], x_offset:x_offset+top_view.shape[1]] = top_view

            # save every frame as png
            # frame = skimage.transform.rotate(frame, 1, clip=True, preserve_range=True)
            # cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_horizon/%s.png'%(file, frame_nr_int), frame)
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/%s_processed/%s.png'%(file, frame_nr_int), frame)
            # quit()
            print(frame_nr_int)
            cv.imshow(file, frame)
            frame_nr_int += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            print(frame_nr_int)
            # pitch_df = pd.DataFrame(data=pitch_angle, columns=['Milliseconds', 'Pitch_from_vis'])
            # print(csv_path + file + '-pitch.csv')
            # pitch_df.to_csv(csv_path + file + '-pitch.csv')
            quit()

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__=='__main__':
    main()
