import numpy as np
import pandas as pd
from sympy import Point3D, Circle
from skimage.draw import circle_perimeter, line
import matplotlib.pyplot as plt
import cv2 as cv
from cordFrames import get_cordFrames
'''
Draws a curve in 2D and projects points to a screen

Implements camera projection as described here:
https://robotacademy.net.au/masterclass/the-geometry-of-image-formation/?lesson=777
https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points
circle in (world) coordinate frame. Origin lies where front wheel touches the street.
unit: m
circle lies on a plane --> Y=0 in all points
'''

# actual focal length = equivalent focal length / crop factor
f = np.float64(0.019/5.6)

# sensor size / video resolution
pu = np.float64(0.0062/1920)
pv = np.float64(0.0045/1080)
# central coordinates of video
u0 = 960
v0 = 540

def draw_curve(radius, cam_frame):
    # radius = 2000
    # a radius with a radius of inf/-inf is a line. Here, threshold is 1000m for speed-up
    if np.abs(radius) > 1000:
        curve2d = line(0,1,0,3000)
    else:
        # calculate circle perimeter in 2d (function gives tuple of arrays: ([x-cord0, x-cord1, ...], [z-cord0, zcord1,...]))
        # One point every 10 cm
        a, b = np.array(circle_perimeter(np.int(radius*10), 0, np.abs(np.int(radius*10))))


        a2 = []# np.zeros((np.int(a.shape[0]/2)-1))
        b2 = []# np.zeros((np.int(a.shape[0]/2)-1))

        # eliminate points behind camera
        for x in range(a.shape[0]):
            if b[x]>0:
                # eliminate points that lie "far away"
                # if np.linalg.norm(a[x]-b[x])<1000:
                a2.append(a[x])
                b2.append(b[x])
        curve2d = (np.asarray(a2), np.asarray(b2))
    # print(curve2d)

    # extend to 4d (homogeneous) with y = 0, w = 1
    curve2d = (curve2d[0]/10, curve2d[1]/10)
    curve4d = (curve2d[0]/10, np.zeros(curve2d[0].shape, dtype=np.int64),curve2d[1]/10, np.ones(curve2d[0].shape, dtype=np.int64))
    curve4d = np.column_stack(curve4d).astype(np.float64)

    # now: array of 4-dim array: [[x0, 0, z0, 1], [x1, 0, z1, 1], ...]

    # print(curve4d)

    # camera rotation inverse (from parent(world) to camera), homogeneous
    cam_rot = cam_frame.transToWorld.rot.inv().as_matrix()
    # print(cam_rot)
    homog_rot = np.identity(cam_rot.shape[0]+1)
    homog_rot[:-1,:-1] = cam_rot
    homog_rot[-1,-1] = 1

    # homogeneous camera translation
    cam_trans = cam_frame.transToWorld.trans
    homog_pos = np.identity(4)
    homog_pos[:-1, -1] = cam_trans

    # print(homog_pos)
    # print(homog_rot)

    # combine trans and rot
    trans_rot = np.dot(homog_pos, homog_rot)
    trans_rot = trans_rot

    # print(trans_rot)

    # matrices for camera intrinsics
    pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
    focal_mat = np.array([[f,0,0,0],[0,f,0,0], [0,0,1,0]], dtype=np.float64)

    # put all together --> camera matrix C
    K = np.dot(pixel_mat, focal_mat)
    C = np.dot(K, trans_rot)

    # print(K)
    # print(C)

    curve_proj = np.column_stack(np.zeros_like(curve2d)).astype(np.float64)

    # multiply every point of curve with camera matrix, divide by last digit to bring to 2D
    for i in range(curve4d.shape[0]):

        u,v,w = np.dot(C, curve4d[i])
        # print(u,v,w)
        curve_proj[i] = u/w, v/w

        # withour matrices kind of like this:
        # x = (f * X / -Z
        # u = x/pu + 960
        #
        # y = (f * Y / -Z
        # v = y/pv + 540


    # check, wether curve has reasonable coordinates
    # print(np.amin(curve_proj[0, :]))
    # print(np.amax(curve_proj[0, :]))
    # print(np.mean(curve_proj[0, :]))
    #
    # print(np.amin(curve_proj[1, :]))
    # print(np.amax(curve_proj[1, :]))
    # print(np.mean(curve_proj[1, :]))

    # print(curve_proj)
    return curve_proj, bird_view(curve2d, trans_rot)

# plot curve without camera perspective (from above, looking at world frame origin)
# also vector describing camera-z-axis (to back of motorcycle)
def bird_view(curve2d, trans_rot):
    # print(trans_rot)
    # vector in worldframe
    look_dir = [[0,0,0,1],[0,0,50,1]]
    homog_look_dir = np.dot(trans_rot, look_dir[0]), np.dot(trans_rot, look_dir[1])
    cam_z = homog_look_dir[0][0],homog_look_dir[0][2],homog_look_dir[1][0],homog_look_dir[1][2]

    my_dpi=96
    fig = plt.figure(figsize=(320/my_dpi, 240/my_dpi), dpi=my_dpi)
    sc = fig.add_subplot(111)
    sc.arrow(*cam_z, head_width=5, head_length=7, fc='blue',ec='black')
    sc.scatter(curve2d[0],curve2d[1], marker='x', color='red', label='time in ms')
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
    _, curve2d = draw_curve(-20, cam_frame)
    # print(curve2d)
