import numpy as np
import pandas as pd
from sympy import Point3D, Circle
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt
import cv2 as cv

# kreis in koordinatensystem. ursprung: schnittpunkt motorrad, ebene (stuetzpunkt)
# einheit: cm
# kreis liegt auf ebene, also Y-Koordinate konstant 0

# zu bild Koordinaten:
# bildX = f * ()

# actual focal length = equivalent focal length / crop factor
# 19/5.6 =

f = np.float64(1.9/5.6)
AoV_hor = 1.476549
AoV_ver = 1.181588
z_far = 1000
z_near = 50


def draw_curve(radius, cam_frame):
    # get curve radius in m, convert to cm
    # radius = 20
    radius *= 10
    # calculate circle perimeter in 2d, extend to 3d(4d) with y-values=0 (1 for 4th dim)
    curve2 = np.array(circle_perimeter(np.int(radius), 0, np.abs(np.int(radius))))
    # eliminate coords behind camera
    a = curve2[0].tolist()
    b = curve2[1].tolist()
    for x in range(len(a)-1, -1, -1):
        if a[x]<=0 or b[x]<=0:
            del(a[x])
            del(b[x])

    # print(a)
    # print(b)
    curve2 = (np.asarray(a), np.asarray(b))

    curve3d = (curve2[0], np.zeros(curve2[0].shape, dtype=np.int64),curve2[1], np.ones(curve2[0].shape, dtype=np.int64))
    curve3d = np.column_stack(curve3d).astype(np.float64)
    # print(curve3d)

    # camera rotation inverse (from parent(world) to camera), homogeneous
    cam_mat = cam_frame.transToWorld.rotQuat.inv().as_matrix()
    homog_rot = np.zeros((cam_mat.shape[0]+1, cam_mat.shape[1]+1))
    # print(homog_rot)
    homog_rot[:-1,:-1] = cam_mat
    homog_rot[-1,-1] = 1

    # homogeneous camera translation
    cam_trans = cam_frame.transToWorld.trans
    homog_pos = np.identity(4)
    homog_pos[:-1, -1] =- cam_trans

    # combine trans and rot
    trans_rot = np.dot(homog_rot, homog_pos)
    # print(trans_rot)
    # curve_homog = np.zeros((curve3d.shape[0], curve3d.shape[1]))
    curve_cam = np.zeros_like(curve3d)
    curve_proj = np.column_stack(np.zeros_like(curve2)).astype(np.float64)

    # perp_proj_mat = np.zeros((4,4), dtype=np.float64)
    # perp_proj_mat[0,0] = np.arctan(AoV_hor/2)
    # perp_proj_mat[1,1] = np.arctan(AoV_ver/2)
    # perp_proj_mat[2,2] = 2/(z_far-z_near)
    # perp_proj_mat[2,3] = (z_far+z_near)/(z_far-z_near)
    # perp_proj_mat[3,2] = -1
    # print(perp_proj_mat)


    for i in range(curve3d.shape[0]):
        # print(homog_rot[i])
        # print(homog_pos[i])

        # curve_homog[i] = np.dot(trans_rot, curve3d[i])

        curve_cam[i] = np.dot(trans_rot, curve3d[i])

        # view = np.dot(curve_cam[i], perp_proj_mat)
        # view = np.dot(perp_proj_mat, curve_cam[i])

        # print(view)

        # image space :
        # x = view[0]/view[3]
        # y = view[1]/view[3]
        # z = view[2]/view[3]
        x = f * (curve_cam[i][0]/-curve_cam[i][2]) + 0.617/2
        y = f * (curve_cam[i][1]/-curve_cam[i][2]) + 0.455/2

        x_norm = (x+1)/2
        y_norm = (y+1)/2

        x_rast = np.floor(x_norm * 1920)
        y_rast = np.floor(y_norm * 1080)
        curve_proj[i] = x_rast,y_rast
        # calc pos on 'film' of camera, discard everythin behind camera.
        # if np.greater(curve_cam[i,2],0):
        #     x_film = f * (curve_cam[i,0]/curve_cam[i,3]) + 0.617/2
        #     y_film = f * (curve_cam[i,1]/curve_cam[i,3]) + 0.455/2
        #     # curve_proj[i] = x_film*100, y_film*100
        #     curve_proj[i] = x*100, y*100

    # print(np.amin(curve_proj[0, :]))
    # print(np.amax(curve_proj[0, :]))
    # print(np.mean(curve_proj[0, :]))
    #
    # print(np.amin(curve_proj[1, :]))
    # print(np.amax(curve_proj[1, :]))
    # print(np.mean(curve_proj[1, :]))

    # print(curve_proj))
    return curve_proj
