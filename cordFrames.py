import numpy as np
from scipy.spatial.transform import Rotation as R
# from abc import ABC, abstractmethod

class transform():
    def __init__(self, trans, rotQuat):
        self.trans = trans
        self.rot = R.from_quat(rotQuat)
    def calc_new_transform(self, transform1):
        return transform(self.trans + transform1.trans, (self.rot*transform1.rot).as_quat())


# world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road
# +X: right, +Y: Up, +Z TO the back of motorcycle
class worldFrame():
    def __init__(self):
        self.name = 'world'
        self.transToWorld = self.transToParent = transform([0, 0, 0], [0, 0, 0, 1])
        self.children = []
        self.parent = None
    def __str__(self):
        return 'world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road \n+X: right, +Y: Up, +Z TO the back of motorcycle'

class cordFrame():
    def __init__(self, name, data, world_frame, parent, children, transToParent):
        self.data = data
        self.name = name
        self.parent = world_frame if parent is None else parent
        self.parent.children.append(self)
        self.children = children
        self.transToParent = transToParent


        # TODO: direction of matmul?
        self.transToWorld = self.parent.transToWorld.calc_new_transform(self.transToParent)

    def update(self, newData, newTransToParent):
        if newData is not None:
            self.data = newData
        if transToParent is not None:
            self.transToParent = newTransToParent

    def __str__(self):
        return 'Coordinate Frame for {}.\nTranslation to world frame:\n{}\nOrientation:\n{}'.format(self.name, self.transToWorld.trans, self.transToWorld.rot.as_quat())

def get_cordFrames():
    # world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road
    # +X: left(!), +Y: Up, +Z TO the front (!) of motorcycle
    world_frame = worldFrame()
    # plane coordinate frame
    # same as world
    plane_frame = cordFrame(data=None,
                            name='plane',
                            world_frame=world_frame,
                            parent=world_frame,
                            children=[],
                            transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
                                                             [0, 0, 0, 1])
                            )


    # imu coordinate frame
    # lies 121,37 cm above origin of world frame
    # +X: up (away from world origin), +Y: to the left(? TODO!!!) +Z: to the back of motorcycle

    imu_frame = cordFrame(data=None,
                          name='IMU',
                          world_frame=world_frame,
                          parent=world_frame, children=[],
                          transToParent=transform(np.array([0, -1.2137, 0], dtype=np.float64),
                                                  [[np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)],
                                                  [0, 0, np.sin(-np.pi/4),np.cos(-np.pi/4)]])

                         )


    # camera coordinate frame
    # position same as IMU
    # orientation as a camera: +X: right (width), +Y: Up (to worldframe, height), +Z to the back of motorcycle

    cam_frame = cordFrame(data=None,
                          name='Camera',
                          world_frame=world_frame,
                          parent=world_frame,
                          children=[],
                          transToParent=transform(np.array([0, -1.2137, 0], dtype=np.float64),
                                                  [0, np.sin(np.pi/2), 0, np.cos(np.pi/2)])
                          )

    # print(world_frame)
    # print(plane_frame)
    # print(imu_frame)
    # print(cam_frame)

    return world_frame, plane_frame, cam_frame
