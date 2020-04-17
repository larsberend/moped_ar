import numpy as np
import pandas as pd
from sympy import Point3D, Circle
from skimage.draw import circle_perimeter, line
import matplotlib.pyplot as plt
import cv2 as cv
from cordFrames import get_cordFrames

# kreis in koordinatensystem. ursprung: schnittpunkt motorrad, ebene (stuetzpunkt)
# einheit: cm
# kreis liegt auf ebene, also Y-Koordinate konstant 0

# zu bild Koordinaten:
# bildX = f * ()

# actual focal length = equivalent focal length / crop factor
# 19/5.6 =

f = np.float64(0.035/5.6)
# sensor size
pu = np.float64(0.0062/1920)
pv = np.float64(0.0045/1080)

AoV_hor = 1.476549
AoV_ver = 1.181588
z_far = 20
z_near = 1


def draw_curve(radius, cam_frame):
    # get curve radius in m, convert to cm
    if np.abs(radius) > 300:
        curve2 = line(0,1,0,300)
        # print('line')
    else:
        # radius *= 10
        # print(radius)
        # calculate circle perimeter in 2d
        curve2 = np.array(circle_perimeter(np.int(radius), 0, np.abs(np.int(radius))))

        # eliminate coords behind camera
        a, b = curve2
        a2 = []# np.zeros((np.int(a.shape[0]/2)-1))
        b2 = []# np.zeros((np.int(a.shape[0]/2)-1))

        for x in range(a.shape[0]):
            if b[x]>=0:
                a2.append(a[x])
                b2.append(b[x])
        curve2 = (np.asarray(a2), np.asarray(b2))
    print(curve2)
    # extend to 3d(4d) with y-values=0 (1 for 4th dim)
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
    print(trans_rot)
    # curve_homog = np.zeros((curve3d.shape[0], curve3d.shape[1]))
    curve_cam = np.zeros_like(curve3d)
    curve_proj = np.column_stack(np.zeros_like(curve2)).astype(np.float64)

    # perp_proj_mat = np.zeros((4,4), dtype=np.float64)
    # perp_proj_mat[0,0] = np.arctan(AoV_hor/2)
    # perp_proj_mat[1,1] = np.arctan(AoV_ver/2)
    # # perp_proj_mat[2,2] = -1
    # # perp_proj_mat[3,2] = -1
    # perp_proj_mat[2,2] = -z_far/(z_far-z_near)
    # perp_proj_mat[3,2] = -(z_far*z_near)/(z_far-z_near)
    # perp_proj_mat[2,3] = -1

    # print(perp_proj_mat)




    for i in range(curve3d.shape[0]):
        # print(homog_rot[i])
        # print(homog_pos[i])

        # curve_homog[i] = np.dot(trans_rot, curve3d[i])

        curve_cam[i] = np.dot(trans_rot, curve3d[i])
        print('curve_cam')
        print(curve_cam[i])
        # view = np.dot(curve_cam[i], perp_proj_mat)
        # view = view[:3] / view[3]
        # print('view')
        # print(view)

        x = (f * curve_cam[i][0]) / -curve_cam[i][2]
        u = x/pu + 960

        y = (f * curve_cam[i][1]) / -curve_cam[i][2]
        v = y/pv + 540
        print('x,y')
        print((x,y))
        print('u,v')
        print((u,v))
        # y = f * (curve_cam[i][1]/-curve_cam[i][2]) + 0.0455/2

        # x = f * (curve_cam[i][0]/-curve_cam[i][2])
        # y = f * (curve_cam[i][1]/-curve_cam[i][2])
        # x,y = view[:2]
        # print((x,y))
        # curve_proj[i] = x,y
        # # print(x,y)
        #
        # x_norm = (x+1)/2
        # y_norm = (y+1)/2
        #
        # x_rast = np.floor(x_norm * 1920)
        # y_rast = np.floor(y_norm * 1080)
        # print((y, y_norm, y_rast))
        # curve_proj[i] = x_rast, y_rast
        curve_proj[i] = u, v
        # calc pos on 'film' of camera, discard everythin behind camera.
        # if np.greater(curve_cam[i,2],0):
        #     x_film = f * (curve_cam[i,0]/curve_cam[i,3]) + 0.617/2
        #     y_film = f * (curve_cam[i,1]/curve_cam[i,3]) + 0.455/2


    # print(np.amin(curve_proj[0, :]))
    # print(np.amax(curve_proj[0, :]))
    # print(np.mean(curve_proj[0, :]))
    #
    # print(np.amin(curve_proj[1, :]))
    # print(np.amax(curve_proj[1, :]))
    # print(np.mean(curve_proj[1, :]))

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
