import numpy as np
from scipy.spatial.transform import Rotation as R
# from abc import ABC, abstractmethod

class transform():
    def __init__(self, trans, rotQuat):
        self.trans = trans
        self.rotQuat = rotQuat
    def calc_new_transform(self, transform1):
        return transform(self.trans + transform1.trans, self.rotQuat * transform1.rotQuat)


# world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road
# +X: right, +Y: Up, +Z TO the back of motorcycle
class worldFrame():
    def __init__(self):
        self.name = 'world'
        self.transToWorld = self.transToParent = transform([0, 0, 0], R.from_quat([0, 0, 0, 1]))
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
        return 'Coordinate Frame for {}.\nPosition relative to world frame:\n{}\nOrientation:\n{}'.format(self.name, self.transToWorld.trans, self.transToWorld.rotQuat.as_quat())
    #
    # def getParent(self):
    #     return self.parent
    #
    # def getChildren(self):
    #     return self.chrildren

def get_cordFrames():
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
                                                  R.from_quat(np.array([[np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)],
                                                                        [0, 0, np.sin(-np.pi/4),np.cos(-np.pi/4)]
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

    return world_frame, plane_frame, cam_frame
