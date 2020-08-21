import numpy as np
import pandas as pd
from sympy import Point3D, Circle
from skimage.draw import circle_perimeter, line
import matplotlib.pyplot as plt
import cv2 as cv
from cordFrames import get_cordFrames
# from birdview import point_warp, get_homography2
from scipy.spatial.transform import Rotation as R
from camera_K import K_homog as K

'''
Draws a curve in 2D and projects points to a screen

Implements camera projection as described here:
https://robotacademy.net.au/masterclass/the-geometry-of-image-formation/?lesson=777
https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points
circle in (world) coordinate frame. Origin lies where front wheel touches the street / mass center of motorcycle on street.
unit: m
circle lies on a plane --> Y=0 in all points
'''


factor = 1 # take 100 points per meter, show 1
view_dist = 100
# factor = 10 # take 100 points per meter, show 10
# view_dist = 100
grav_center = 2.805 # behind lowest point visible in calib image

calib_pitch_from_vis = 0.698131700797732

# get points which lie on curve in a plane
def get_arc_points(radius, view_dist, factor):
    # radius = 10
    z_values = np.arange(0, view_dist, 1/factor)
    x_values = np.sqrt(radius**2 - z_values**2) - np.abs(radius)

    if radius > 0:
        x_values *= -1

    return x_values , z_values

# get points which lie on a plane and in image of a curve
def draw_curve(radius, cam_frame, mc_frame, pitch, horizon=False):

    if not horizon:
        x,z = get_arc_points(radius, view_dist, factor)

    else:
        hori_x = np.arange(-100000, 100000, 1000)
        hori_z = np.full_like(hori_x, fill_value=4000)

        x = hori_x
        z = hori_z
    '''
    # to check warping: warp birdview back to regular view (buggy)
    if pitch is not None:
        print((x,z))
        pitch_rot = R.from_euler('xyz', [0, pitch, 0], degrees=False).as_matrix()
        H = get_homography2(pitch_rot, K)
        Hinv = np.linalg.inv(H)
        curve_proj = point_warp((x, z), Hinv, np.zeros((10000, 10000)))
        print(x.shape)
        print(z.shape)
        top = top_view((x,z), 1, radius)
        curve_proj = np.column_stack((curve_proj[0], -curve_proj[1])).astype(np.float64)
        return curve_proj, top
    '''

    curve4d = np.column_stack((x, np.zeros(x.shape), z, np.ones(x.shape))).astype(np.float64)
    # show only 1 point per m
    curve4d = curve4d[(curve4d[:,2]%1)==0]

    # now: array of 4-dim array: [[x0, 0, z0, 1], [x1, 0, z1, 1], ...]
    # print(curve4d[:5])

    #  camera rotation inverse (from world to camera)
    world_to_cam_rot = cam_frame.transToWorld.rot.inv()#.as_matrix()
    # print('to world and back:')
    # print(world_to_cam_rot.as_matrix())

    '''
    # front wheel to world rotation
    fw_to_world_rot = fw_frame.transToWorld.rot#.as_matrix()

    # camera to front wheel
    cam_rot = world_to_cam_rot * fw_to_world_rot
    cam_rot = cam_rot.as_matrix()
    '''

    # rotation from mass center to camera
    mc_to_world_rot = mc_frame.transToWorld.rot
    # print('mc')
    # print(mc_to_world_rot.as_matrix())
    cam_rot = world_to_cam_rot * mc_to_world_rot
    cam_rot = cam_rot.as_matrix()


    homog_rot = np.identity(cam_rot.shape[0]+1)
    homog_rot[:-1,:-1] = cam_rot
    homog_rot[-1,-1] = 1
    print('cam_rot')
    print(homog_rot)

    # homogeneous camera translation
    # cam_trans = cam_frame.transToWorld.trans - fw_frame.transToWorld.trans
    cam_trans = cam_frame.transToWorld.trans - mc_frame.transToWorld.trans
    homog_pos = np.identity(4)
    homog_pos[:-1, -1] = -cam_trans

    # combine trans and rot
    trans_rot = np.dot(homog_pos, homog_rot)
    # print(trans_rot)

    C = np.dot(K, trans_rot)
    # print('K, C')
    # print(K)
    # print(C)
    # print(curve4d.shape[0])

    curve_proj = np.zeros((curve4d.shape[0],2),dtype=np.float64)

    # multiply every point of curve with camera matrix, divide by last digit to bring to 2D pixel coords
    for i in range(curve4d.shape[0]):

        u,v,w = np.dot(C, curve4d[i])
        # print(u,v,w)
        curve_proj[i] = u/w, v/w

        # without matrices kind of like this:
        # x = (f * X) / -Z
        # u = x/pu + 960
        #
        # y = (f * Y) / -Z
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

    return curve_proj, top_view((x,z), trans_rot, radius)


# plot curve without camera perspective (from above, looking at world frame origin)
# also vector describing camera-z-axis (to back of motorcycle)
def top_view(curve2d, trans_rot, radius):
    rr,cc = circle_perimeter(int(radius), 0, int(np.abs(radius)))
    rr = rr[cc>0]
    cc = cc[cc>0]
    # print((rr,cc))
    # quit()
    my_dpi=96
    fig = plt.figure(figsize=(320/my_dpi, 240/my_dpi), dpi=my_dpi)
    sc = fig.add_subplot(111)
    # sc.arrow(*cam_z, head_width=5, head_length=7, fc='blue',ec='black')
    sc.scatter(rr,cc, marker='x', color='red', label='time in ms')
    # sc.scatter(curve2d[0],curve2d[1], marker='x', color='red', label='time in ms')
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
    world_frame, fw_frame, cam_frame, imu_frame = get_cordFrames()
    curve2d, top = draw_curve(-50, cam_frame, fw_frame, np.radians(80))
    print(curve2d)
