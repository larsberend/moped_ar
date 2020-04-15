import numpy as np
import pandas as pd
from cordFrames import cordFrame, worldFrame, transform
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from skimage.draw import circle_perimeter, line
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick_wenigSelf import CMad


def main():
    # world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road
    # +X: left(!), +Y: Up, +Z TO the front (!) of motorcycle
    world_frame = worldFrame()
    # print(worldFrame.name)
    # plane coordinate frame
    # same as world
    plane_frame = cordFrame(data=None,
                            name='plane',
                            world_frame=world_frame,
                            parent=world_frame,
                            children=[],
                            transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
                                          R.from_quat(np.array([0, 0, 0, 1], dtype=np.float64)))
                            )

    # imu coordinate frame
    # lies 121,37 cm above origin of world frame
    # +X: up (away from world origin), +Y: to the left(? TODO!!!) +Z: to the back of motorcycle

    imu_frame = cordFrame(data=None,
                          name='IMU',
                          world_frame=world_frame,
                          parent=world_frame, children=[],
                          transToParent=transform(np.array([0, -121.37, 0], dtype=np.float64),
                                                  R.from_quat(np.array([[0, 0, np.sin(np.pi/4),np.cos(np.pi/4)],
                                                              [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)]
                                                             ], dtype=np.float64))
                                                 )
                        )

    # blau rot vertauscht
    # camera coordinate frame
    # position same as IMU
    # orientation as a camera: +X: right (width), +Y: down (to worldframe, height), +Z in driving direction

    cam_frame = cordFrame(data=None,
                          name='Camera',
                          world_frame=world_frame,
                          parent=world_frame,
                          children=[],
                          transToParent=transform(np.array([0, -121.37, 0], dtype=np.float64),
                          R.from_quat(np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)], dtype=np.float64)))
                          )

    # img_frame = cordFrame(data=None,
    #                         name='Image',
    #                         world_frame=world_frame,
    #                         parent=cam_frame,
    #                         children=[],
    #                         transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
    #                         R.from_quat(np.array([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)], dtype=np.float64)))
    #
    #                         )

    # print(world_frame)
    # print(plane_frame)
    # print(imu_frame)
    # print(cam_frame)

    csv_path = '../100GOPRO/kandidaten/csv/'
    video_path = '../100GOPRO/kandidaten/'
    file = '3_2'
    test = turn()
    cmad = CMad()

    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwick.csv')[['Milliseconds','Radius']].to_numpy()[2:].swapaxes(1,0)
    # print(radius_madgwick)
    # print(video_path+'.mp4')
    radius_madgwick[0] -= radius_madgwick[0,0]
    cap = cv.VideoCapture(video_path + file + '.mp4')
    # print(cap.isOpened())
    current = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        nearest_rad = find_nearest(radius_madgwick[0], cap.get(0))
        # print(radius_madgwick[0])
        # print(cap.get(0))
        # print(nearest_rad)
        # print(radius_madgwick[0,nearest_rad])
        radius = radius_madgwick[1,nearest_rad]
        # print(radius)
        curve_proj = draw_curve(radius, cam_frame)

        for j in curve_proj.astype(np.int32):
            cv.circle(frame, (j[0], j[1]), 5, (0,255,0))

        # Display the resulting frame
        cv.imshow('frame', frame)
        # current += 4
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return idx
















if __name__=='__main__':
    main()
