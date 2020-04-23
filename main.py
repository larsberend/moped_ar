import numpy as np
import pandas as pd
from cordFrames import cordFrame, worldFrame, transform, get_cordFrames
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from skimage.draw import circle_perimeter, line
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick import CMad

'''
Plays frames from a video file and draws trajectory of the motorcycle in each of them.
Updates camera-parameters with data from IMU.
'''

def main():

    world_frame, plane_frame, cam_frame, imu_frame = get_cordFrames()

    csv_path = '../100GOPRO/kandidaten/csv/'
    video_path = '../100GOPRO/kandidaten/'
    file = '2_3'
    start_msec = 2000.0
    font = cv.FONT_HERSHEY_SIMPLEX

    # get data from csv as array, ignore first two elements to resolve empty datapoints
    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwickQuat.csv')[['Milliseconds','Radius','Quat']].to_numpy()[2:].swapaxes(1,0)
    # print(radius_madgwick)

    # radius_madgwick[0] -= radius_madgwick[0,0]

    cap = cv.VideoCapture(video_path + file + '.mp4')
    # out = cv.VideoWriter('3_2_curve.avi', cv.VideoWriter_fourcc('M','J','P','G'), 59.94, (1920,1080))
    cap.set(0, start_msec)
    # int counting frames(for saving snapshots)
    frame_nr_int = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            height,width = frame.shape[:2]
            # find closest Datapoint in time to current Frame
            nearest_rad = find_nearest(radius_madgwick[0], cap.get(0))
            mil, radius, ori = radius_madgwick[:,nearest_rad]
            # cast Quat from string(pandas) to float
            if type(ori)==str:
                ori = [np.float64(x.strip(' []')) for x in ori.split(',')]
            # rearrange to scalar-last format, cast to Euler to extract only Roll-Angle
            ori = [ori[1],ori[2],ori[3],ori[0]]
            ori_eul = R.from_quat(ori).as_euler('xyz',degrees=False)
            ori_yz = R.from_euler('xyz',[0, 0, -ori_eul[2]],degrees=False).as_quat()

            # check, if Euler conversion returns same Quaternion
            # assert np.abs(np.dot(R.from_euler('xyz', ori_eul).as_quat(), ori) - 1) < 0.00001

            # get position of IMU by calculating circular cord with height above world and rotation around rolling axis
            transl = 2 * 1.2137 * np.sin(ori_yz[2])

            # calc new transform of IMU to world
            imu_trans = transform([transl, -1.2137, 0], [
                                                    [0, 0, np.sin(np.pi/4),np.cos(np.pi/4)],
                                                    [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)],
                                                    # [ori_yz[1], 0, 0, np.cos(np.arcsin(ori_yz[1]))]
                                                    ori_yz
                                                    # [0,np.sin(-np.pi/32),0,np.cos(-np.pi/32)]
                                                   ]
                                 )
            imu_frame.update_ori(imu_trans)


            cv.putText(frame, 'Frame Nr: %s'%(cap.get(0)), (1650,20), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'Radius: %s'%(radius), (1650,40), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'MIlliseconds(IMU): %s'%(mil), (1650,60), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)

            # calc visible trajectory in frame and view from above
            curve_proj, bird_view = draw_curve(radius, cam_frame)

            for j in curve_proj.astype(np.int32):
                if j[0]<1920 and j[1]<1080:
                    cv.circle(frame, (j[0], j[1]), 5, (0,255,0))

            x_offset = width - bird_view.shape[1]
            y_offset = np.int(height/3) - bird_view.shape[0]
            frame[y_offset:y_offset+bird_view.shape[0], x_offset:x_offset+bird_view.shape[1]] = bird_view

            # save every frame as png
            # cv.imwrite('../100GOPRO/kandidaten/%s_processed/%s.png'%(file, frame_nr_int), frame)
            frame_nr_int += 1

            cv.imshow(file, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print(frame_nr_int)
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


# return index of value closest to input vale
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return idx





if __name__=='__main__':
    main()
