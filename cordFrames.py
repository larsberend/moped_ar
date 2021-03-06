import numpy as np
from scipy.spatial.transform import Rotation as R
'''
characterize a transform in 3D (translation and rotation)
initialize with 3D vector and Quaternion(s)
'''
class transform():
    def __init__(self, trans, rotQuat):
        trans = np.array(trans,dtype=np.float64)
        rotQuat = np.array(rotQuat,dtype=np.float64)
        self.trans = trans
        if len(rotQuat.shape)<2:
            self.rot = R.from_quat(rotQuat)
        else:
            self.rot = R.from_quat([0,0,0,1])
            for rota in rotQuat:
                self.rot *= R.from_quat(rota)


    def calc_new_transform(self, transform1):
        return transform(self.trans + transform1.trans, (self.rot*transform1.rot).as_quat())

'''
world coordinate frame, so far: origin at ground level of world-plane, contact point of front-wheel and road
+X: left, +Y: Up, +Z driving direction
'''

class worldFrame():
    def __init__(self):
        self.name = 'world'
        self.transToWorld = self.transToParent = transform([0, 0, 0], [0, 0, 0, 1])
        self.children = []
        self.parent = None
    def __str__(self):
        return 'world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road \n+X: left, +Y: Up, +Z: driving direction'


'''
class for coordinate frames located in world space.
use transform class to show position(translation, rotation) relative to world origin
'''
class cordFrame():
    def __init__(self, name, data, world_frame, parent, children, transToParent):
        self.data = data
        self.name = name
        self.parent = world_frame if parent is None else parent
        self.parent.children.append(self)
        self.children = children
        self.transToParent = transToParent

        self.__calcTransToWorld__()

    def update_ori(self, transToParent):
        self.transToParent = transToParent
        self.__calcTransToWorld__()
        for child in self.children:
            child.__calcTransToWorld__()

    # returns transform to world frame, irrespective of child or grandchild of world
    def __calcTransToWorld__(self):
        self.transToWorld = self.parent.transToWorld.calc_new_transform(self.transToParent)


    def __str__(self):
        return 'Coordinate Frame for {}.\nTranslation to world frame:\n{}\nOrientation:\n{}'.format(self.name, self.transToWorld.trans, self.transToWorld.rot.as_quat())

'''
Get all coordinate frames intended for moped-ar with this function
'''
def get_cordFrames():
    world_frame = worldFrame()

    # front wheel coordinate frame
    # moving coordinate frame, origin at touching point of front wheel and street
    # so far: same as world
    fw_frame = cordFrame(data=None,
                            name='Front Wheel',
                            world_frame=world_frame,
                            parent=world_frame,
                            children=[],
                            transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
                                                             [0, 0, 0, 1])
                            )


    # imu coordinate frame
    # lies 1.2137 m above origin of world frame
    # +X: up (away from world origin), +Y: to the left(?) +Z: to the back of motorcycle
    imu_frame = cordFrame(data=None,
                          name='IMU',
                          world_frame=world_frame,
                          parent=fw_frame,
                          children=[],
                          transToParent=transform(np.array([1.2137,0, 0], dtype=np.float64),
                                                  [
                                                  # [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)],
                                                  [0, 0, np.sin(np.pi/4),np.cos(np.pi/4)],
                                                  [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)]
                                                  ])

                         )


    # camera coordinate frame
    # position same as IMU
    # orientation as a camera: +X: right (width), +Y: Up, +Z to the back of motorcycle
    # parent: IMU-sensor, quarter positive turn around z-axis
    cam_frame = cordFrame(data=None,
                          name='Camera',
                          world_frame=world_frame,
                          parent=imu_frame,
                          children=[],
                          transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
                                                  [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
                          )

    mc_frame = cordFrame(data=None,
                         name='Mass center on ground',
                         world_frame=world_frame,
                         parent=fw_frame,
                         children=[],
                         transToParent=transform(np.array([0, 0, -0.7105], dtype=np.float64),
                                                 [0, 0, 0, 1])
                        )





    # print(world_frame)
    # print(fw_frame)
    # print(imu_frame)
    # print(cam_frame)

    return world_frame, fw_frame, cam_frame, imu_frame, mc_frame
