import numpy as np
import pandas as pd
from cordFrames import cordFrame, worldFrame, transform
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last

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
                          R.from_quat(np.array([np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)], dtype=np.float64)))
                          )


    print(world_frame)
    print(plane_frame)
    print(imu_frame)
    print(cam_frame)

if __name__=='__main__':
    main()
