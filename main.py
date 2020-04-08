import numpy as np
import pandas as pd
from cordFrames import cordFrame, worldFrame, transform
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from skimage.draw import circle_perimeter

f = np.float64(1.9/5.6)

def main():
    # world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road
    # +X: right, +Y: Up, +Z TO the back of motorcycle
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
    # +X: up (away from world origin), +Y: to the right(? TODO!!!) +Z: to the back of motorcycle
    imu_frame = cordFrame(data=None,
                          name='IMU',
                          world_frame=world_frame,
                          parent=world_frame, children=[],
                          transToParent=transform(np.array([0, 121.37, 0], dtype=np.float64),
                                        R.from_quat(np.array([0, 0, np.sin(np.pi/4),np.cos(np.pi/4)], dtype=np.float64)))
                         )

    # camera coordinate frame
    # position same as IMU
    # orientation as a camera: +X: right (width), +Y: down (to worldframe, height), +Z in driving direction

    cam_frame = cordFrame(data=None,
                          name='Camera',
                          world_frame=world_frame,
                          parent=world_frame,
                          children=[],
                          transToParent=transform(np.array([0, 121.37, 0], dtype=np.float64),
                          R.from_quat(np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)], dtype=np.float64)))
                          )

    image_frame = cordFrame(data=None,
                            name='Image',
                            world_frame=world_frame,
                            parent=cam_frame,
                            children=[],
                            transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
                            R.from_quat(np.array([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)], dtype=np.float64)))

    )
    # print(world_frame)
    # print(plane_frame)
    # print(imu_frame)
    # print(cam_frame)
    radius = 200
    curve2 = circle_perimeter(np.int(radius*10), 0, np.abs(radius))
    curve3d = curve2 + (np.zeros(curve2[0].shape, dtype=np.int64), np.ones(curve2[0].shape, dtype=np.int64))
    curve3d = np.column_stack(curve3d)

    curve_homog = np.zeros((curve3d.shape[0], curve3d.shape[1]))
    cam_mat = cam_frame.transToWorld.rotQuat.as_matrix()
    cam_trans = cam_frame.transToWorld.trans

    homog_rot = np.zeros((cam_mat.shape[0]+1, cam_mat.shape[1]+1))
    # print(homog_rot)
    homog_rot[:-1,:-1] = cam_mat
    homog_rot[-1,-1] = 1

    homog_pos = np.identity(4)
    homog_pos[:-1, -1] =- cam_trans

    curve_proj = np.zeros_like(curve2)

    for i in range(curve3d.shape[0]):
        # print(homog_rot)
        # print(homog_pos)
        trans_rot = np.dot(homog_rot, homog_pos)
        curve_homog[i] = np.dot(trans_rot, curve3d[i])

        x_film = f * (curve_homog[i,0]/curve_homog[i,2])
        y_film = f * (curve_homog[i,1]/curve_homog[i,2])


    print(curve_homog)


















if __name__=='__main__':
    main()
