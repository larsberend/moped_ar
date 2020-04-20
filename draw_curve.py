import numpy as np
import pandas as pd
from sympy import Point3D, Circle
from skimage.draw import circle_perimeter, line
import matplotlib.pyplot as plt
import cv2 as cv
from cordFrames import get_cordFrames

# https://robotacademy.net.au/masterclass/the-geometry-of-image-formation/?lesson=777

# kreis in koordinatensystem. ursprung: schnittpunkt motorrad, ebene (stuetzpunkt)
# einheit: m
# kreis liegt auf ebene, also Y-Koordinate konstant 0

# zu bild Koordinaten:
# bildX = f * ()

# actual focal length = equivalent focal length / crop factor
# 19/5.6 =

f = np.float64(0.019/5.6)
# sensor size
pu = np.float64(0.0062/1920)
pv = np.float64(0.0045/1080)
u0 = 960
v0 = 540

def draw_curve(radius, cam_frame):
    # radius = 10
    if np.abs(radius) > 1000:
        curve2 = line(0,1,0,3000)
        # print('line')
    else:
        # radius *= 10
        # print(radius)
        # calculate circle perimeter in 2d (function gives tuple of arrays: ([x-cord0, x-cord1, ...], [z-cord0, zcord1,...]))
        a, b = np.array(circle_perimeter(np.int(radius*10), 0, np.abs(np.int(radius*10))))


        a2 = []# np.zeros((np.int(a.shape[0]/2)-1))
        b2 = []# np.zeros((np.int(a.shape[0]/2)-1))

        # eliminate points behind camera
        for x in range(a.shape[0]):
            if b[x]>0:
                a2.append(a[x])
                b2.append(b[x])
        curve2 = (np.asarray(a2), np.asarray(b2))
    # print(curve2)
    # extend to 3d(4d) with y-values=0 (1 for 4th dim)
    curve2 = (curve2[0]/10, curve2[1]/10)
    curve3d = (curve2[0]/10, np.zeros(curve2[0].shape, dtype=np.int64),curve2[1]/10, np.ones(curve2[0].shape, dtype=np.int64))
    curve3d = np.column_stack(curve3d).astype(np.float64)

    # now: array of 4-dim array: [[x0, 0, z0, 1], [x1, 0, z1, 1], ...]

    # print(curve3d)

    # camera rotation inverse (from parent(world) to camera), homogeneous
    cam_rot = cam_frame.transToWorld.rot.inv().as_matrix()
    print(cam_rot)
    homog_rot = np.zeros((cam_rot.shape[0]+1, cam_rot.shape[1]+1))
    homog_rot[:-1,:-1] = cam_rot
    homog_rot[-1,-1] = 1

    # homogeneous camera translation
    cam_trans = cam_frame.transToWorld.trans
    homog_pos = np.identity(4)
    homog_pos[:-1, -1] = cam_trans

    # combine trans and rot
    trans_rot = np.dot(homog_rot, homog_pos)
    trans_rot = trans_rot
    # print(trans_rot)
    # matrices for camera intrinsics
    pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
    # wieso -f ?
    focal_mat = np.array([[-f,0,0,0],[0,f,0,0], [0,0,1,0]], dtype=np.float64)

    # put all together --> camera matrix C
    K = np.dot(pixel_mat, focal_mat)
    C = np.dot(K, trans_rot)

    # print(K)
    # print(C)

    curve_proj = np.column_stack(np.zeros_like(curve2)).astype(np.float64)
    for i in range(curve3d.shape[0]):

        u,v,w = np.dot(C, curve3d[i])
        # print(u,v,w)
        curve_proj[i] = u/w, v/w

        # x = (f * curve_cam[i][0]) / -curve_cam[i][2]
        # u = x/pu + 960
        #
        # y = (f * curve_cam[i][1]) / -curve_cam[i][2]
        # v = y/pv + 540



    print(np.amin(curve_proj[0, :]))
    print(np.amax(curve_proj[0, :]))
    print(np.mean(curve_proj[0, :]))

    print(np.amin(curve_proj[1, :]))
    print(np.amax(curve_proj[1, :]))
    print(np.mean(curve_proj[1, :]))

    # print(curve_proj)
    return curve_proj, bird_view(curve2)

def bird_view(curve2):
    my_dpi=96
    fig = plt.figure(figsize=(320/my_dpi, 240/my_dpi), dpi=my_dpi)
    sc = fig.add_subplot(111)
    sc.scatter(curve2[0],curve2[1], marker='x', color='red', label='time in ms')
    sc.set_xlim(-100, 100)
    sc.set_ylim(-100, 100)
    sc.set_title('Birdview of circle in m')
    sc.invert_xaxis()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


if __name__ == '__main__':
    world_frame, plane_frame, cam_frame = get_cordFrames()
    _, curve2 = draw_curve(-20, cam_frame)
    # print(curve2)
