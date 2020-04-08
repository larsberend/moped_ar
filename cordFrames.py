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
