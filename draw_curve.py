import numpy as np
import pandas as pd
from sympy import Point3D, Circle
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt
# kreis in koordinatensystem. ursprung: schnittpunkt motorrad, ebene (stuetzpunkt)
# einheit: cm
# kreis liegt auf ebene, also Y-Koordinate konstant 0

# zu bild Koordinaten:
# bildX = f * ()

# actual focal length = equivalent focal length / crop factor
# 19/5.6 =

class Curve():
    def __init__(self, radius):
        self.c = Circle(Point3D([radius, 0, 0]), radius)


if __name__=='__main__':
    radius = -20
    curve = Curve(radius)
    print(curve.c)

    curve2 = circle_perimeter(np.int(radius*10), 0, np.abs(radius*10))
    curve3d = curve2 + (np.zeros(curve2[0].shape, dtype=np.int64), )
    print('Kurve in world coords')
    print(curve3d)

    curve3d= np.column_stack(curve3d)
    print(curve3d[0])

    plt.scatter(curve2[0], curve2[1])
    plt.savefig('curve.png')
