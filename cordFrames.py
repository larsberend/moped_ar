import numpy as np
from scipy.spatial.transform import Rotation as R

class cordFrame():
    def __init__(self, parent=world, child=None):
        self.parent = parent
        self.child = child
