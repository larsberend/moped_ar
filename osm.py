import math
import numpy as np
import cv2 as cv
from urllib3 import PoolManager
import pandas as pd
from skimage.draw import circle_perimeter
import skimage.transform

R = 6371000

'''
https://www.netzwolf.info/osm/tilebrowser.html?lat=51.157800&lon=6.865500&zoom=14#tile
'''
def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def get_tile(xt, yt, zoom):
    # xt, yt = deg2num(lat, lon, zoom)
    url = 'https://a.tile.openstreetmap.org/%s/%s/%s.png'%(zoom,xt,yt)
    print(url)

    user_agent = {'user-agent': 'motorcycleAR-0.1 contact berend.brandt@gmail.com'}
    http = PoolManager(headers=user_agent)
    response = http.request('GET', url)
    print(response)

    # resp = requests.get(url, data={'user-agent':'research in Motorcycle AR, contact: berend.brandt@gmail.com'})
    # print(resp.content)
    tile = np.asarray(bytearray(response.data), dtype='uint8')
    tile = cv.imdecode(tile, cv.IMREAD_COLOR)
    cv.imwrite('first_tile.png', tile)
    return tile

# def IMU_pos(tile, lat_deg, lon_deg, zoom):
def IMU_pos(lat_deg, lon_deg, zoom):
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)

    lon_rad_merc = lon_rad
    lat_rad_merc = np.log(np.tan(lat_rad) + 1/np.cos(lat_rad))

    x = (1 + lon_rad_merc / np.pi) / 2
    y = (1 - lat_rad_merc / np.pi) / 2

    xt = np.floor(x* 2**zoom)
    yt = np.floor(y* 2**zoom)

    assert (xt,yt) == deg2num(lat_deg, lon_deg, zoom)

    gps_x = np.floor(np.modf(x * 2**zoom)[0] * 256)
    gps_y = np.floor(np.modf(y * 2**zoom)[0] * 256)

    return gps_x, gps_y

def get_map(lat_deg, lon_deg, zoom):
    path = '../100GOPRO/testfahrt_1006/kandidaten/csv/osm_tiles/'
    try:
        tile_table = pd.read_csv(path + 'tiles.csv')[['xt', 'yt', 'zoom']]
    except:
        print('No csv found, create new df.')
        tile_table = pd.DataFrame(columns=['xt', 'yt', 'zoom'])
    xt, yt = deg2num(lat_deg, lon_deg, zoom)

    tnrs = np.array([
                    [xt-1, yt-1, zoom],
                    [xt, yt-1, zoom],
                    [xt+1, yt-1, zoom],
                    [xt-1, yt, zoom],
                    [xt, yt, zoom],
                    [xt+1, yt, zoom],
                    [xt-1, yt+1, zoom],
                    [xt, yt+1, zoom],
                    [xt+1, yt+1, zoom],
                    ])

    tiles = []
    for tnr in tnrs:
        if (tile_table == tnr).all(1).any():
            tiles.append(cv.imread(path + '%s_%s_%s.png'%(tnr[0], tnr[1],zoom)))
        else:
            tile = get_tile(tnr[0], tnr[1], zoom)
            cv.imwrite(path + '%s_%s_%s.png'%(tnr[0], tnr[1],zoom), tile)
            tiles.append(tile)
            tile_table.loc[len(tile_table.index)] = tnr

    tile_table.to_csv(path + 'tiles.csv')

    main_tile = tiles[4]
    gps_x, gps_y = IMU_pos(lat_deg, lon_deg, zoom)
    main_tile[np.int32(gps_y),np.int32(gps_x)] = [0,0,255]

    theight, twidth = main_tile.shape[:2]
    mheight, mwidth = main_tile.shape[0] * 3, main_tile.shape[1] * 3

    map = np.zeros((mheight, mwidth, 3), dtype=np.uint8)

    k = 0
    for i in range(3):
        for j in range(3):
            map[i*theight : (i+1)*theight, j*twidth: (j+1)*twidth] = tiles[k]
            k += 1

    cv.imwrite('map.png', map)
    return map


def map_traj(map, radius, slope, px_m, gps_px):

    # if radius > 0:
    #     slope *= -1
    rad_px = px_m * radius


    gps_x, gps_y = gps_px
    # gps_x, gps_y = np.where(np.all(map == [0,0,255], axis=-1))
    # print((gps_x, gps_y))
    # quit()
    '''
    perp_slope = -slope


    circle_center = (gps_x + dx(rad_px, perp_slope), gps_y + dy(rad_px, perp_slope))
    other_possible_circle_center = (gps_x - dx(rad_px, perp_slope), gps_y - dy(rad_px, perp_slope)) # going the other way
    print('Slope: %s'%(slope))
    print('Radius: %s'%(radius))
    print('circle_center: %s, %s'%circle_center)
    # print('other_possible_circle_center: %s, %s'%other_possible_circle_center)

    rr,cc = circle_perimeter(np.int64(circle_center[0][0]), np.int64(circle_center[1][0]), np.int64(np.abs(rad_px)), shape=map.shape)
    # rr,cc = circle_perimeter(np.int64(other_possible_circle_center[0][0]), np.int64(other_possible_circle_center[1][0]), np.int64(np.abs(rad_px)), shape=map.shape)
    '''

    # slope of line orthogonal to driving direction
    # perp_slope = -1/slope
    # center of curve is radius from motorcycle
    circle_center = (gps_x, gps_y-rad_px)
    print(circle_center)
    rr,cc = circle_perimeter(np.int64(circle_center[0]), np.int64(circle_center[1]), np.int64(np.abs(rad_px)), shape = map.shape)
    map[rr,cc] = [0,0,255]
    # line_thickness = 3
    #
    # rr[rr<line_thickness] = 0
    # # cc[cc<line_thickness] = 0
    # rr[rr>map.shape[0] - line_thickness] = 0
    #
    # cc[cc<line_thickness] = 0
    # cc[cc>map.shape[1] - line_thickness] = 0


    # for i in range(1,line_thickness):
    #     map[rr+i, cc+i] = [0,0,255]
    #     map[rr-i, cc-i] = [0,0,255]

    #
    # # map[min(rr+1, map.shape[0]), min(cc+1, map.shape[1])] = [0,0,255]
    # map[max(rr-1, 0), max(cc-1, 0)] = [0,0,255]
    # cv.circle(map, (np.int32(gps_y[0]), np.int32(gps_x[0])), 5, (0,255,0), thickness=-1)
    cv.circle(map, (np.int32(gps_y), np.int32(gps_x)), 5, (0,255,0), thickness=-1)
    cv.imwrite('traj_map.png', map)
    return map

def get_ul_corner(tx, ty, zoom):
    xpos = tx/(2**zoom)
    ypos = ty/(2**zoom)

    lmerk = (xpos*2-1)*np.pi
    bmerk = (ypos*2-1)*np.pi

    l = lmerk
    b = 2*np.arctan(np.exp(bmerk))-np.pi/2

    return l, b


def get_measures(xt, yt, zoom):
    ul = get_ul_corner(xt, yt, zoom)
    ur = get_ul_corner(xt+1, yt, zoom)
    ll = get_ul_corner(xt, yt+1, zoom)
    lr = get_ul_corner(xt+1, yt+1, zoom)

    les = np.abs(ul[1]-ll[1])*R
    ris = np.abs(ur[1]-lr[1])*R

    ups = np.abs(ul[0]-ur[0]) * np.cos(ul[1]) *R
    los = np.abs(ll[0]-lr[0]) * np.cos(ll[1]) *R

    print('Left Side: %s, Right Side: %s,\nUpper Side: %s, Lower Side: %s' %(les, ris, ups, los))

    length_mean = np.mean([les,ris,ups,los])

    # 256 px ^= lenght_mean m

    px_meter = 256 / length_mean

    return px_meter

def dy(distance, m):
    return m*dx(distance, m)

def dx(distance, m):
    return distance * np.sqrt(1/((m**2)+1))

def rotate_vec(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


if __name__=='__main__':
    '''
    gps = 50.0377006, 9.2961844
    last_gps = 50.0377052, 9.2961994
    radius = 17025.3838755698
    '''
    lat_lon_rad = pd.read_csv('../100GOPRO/testfahrt_1006/kandidaten/csv/' + '3_2' + '-gyroAcclGpsMadgwickQuat.csv')[['Latitude','Longitude','Radius']].dropna().to_numpy().swapaxes(1,0)
    # print(lat_lon_rad)
    # quit()
    zoom = 18
    last_gps = lat_lon_rad[0,0], lat_lon_rad[1,0]
    print(lat_lon_rad.shape)
    # quit()
    for x in np.arange(1, lat_lon_rad.shape[1]):
        print((x, lat_lon_rad.shape[1]))
        gps = lat_lon_rad[0,x], lat_lon_rad[1,x]
        radius = lat_lon_rad[2,x]

        map = get_map(gps[0], gps[1], zoom)
        xt, yt = deg2num(gps[0], gps[1], zoom)
        pixel_meter = get_measures(xt, yt, zoom)

        gps_px = IMU_pos(gps[0], gps[1], zoom)

        last_gps_px = IMU_pos(last_gps[0], last_gps[1], zoom)
        gps_px = gps_px[0]+256, gps_px[1]+256
        last_gps_px = last_gps_px[0]+256, last_gps_px[1]+256

        # print(gps_px)
        # print(last_gps_px)
        #     if radius > 0:
        #         slope_px = -1
        #     else:
        #         slope_px = 1
        # print('slope is at %s now.'%(slope_px))
        # print('radius is %s'%(radius))
        # else:
        '''
        slope = (last_gps[1]-gps[1]) / (last_gps[0]-gps[0])

        if (last_gps_px[0]-gps_px[0]) == 0:
            print('division by zero')
            rotate_angle = 180
        else:
            slope_px = (last_gps_px[1] - gps_px[1]) / (last_gps_px[0] - gps_px[0])
            # rotate_angle =  -(90 + np.abs(np.degrees(0 - np.arctan(slope_px))))
            rotate_angle =  np.degrees(0 - np.arctan(slope_px))
            print(np.degrees(0-np.arctan(slope_px)))
            # quit()
        '''
        slope_px = None
        lat1, lon1 = np.radians(last_gps)
        lat2, lon2 = np.radians(gps)

        dLon = lon2 - lon1
        y = np.sin(dLon) * np.cos(lat2)
        xd = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) *np.cos(dLon)

        bearing = np.arctan2(y, xd) * 180 / np.pi
        print(bearing)
        if (bearing < 0):
            bearing += 360;
        rotate_angle = bearing

        print('Rotate Angle: %s'%(rotate_angle))
        # quit()
        # rotate_angle = 90
        rotated_map = skimage.transform.rotate(map, rotate_angle, resize=False,clip=True, preserve_range=True)

        gps_zero = np.zeros((map.shape[0], map.shape[1], 1))
        gps_zero[np.int32(gps_px[1]), np.int32(gps_px[0])] = 255
        rot_gps_zero = skimage.transform.rotate(gps_zero, rotate_angle, resize=False,clip=True, preserve_range=True)
        gps_rot = np.where(np.all(rot_gps_zero!=[0], axis=-1))
        # print(np.unique(rot_gps_zero, return_counts=True))
        rotated_gps_px = gps_rot[0][0], gps_rot[1][0]

        # cv.imwrite('gps_zero.png', rot_gps_zero)
        # rotated_gps_px = rotate_vec((256+128, 256+128), (gps_px[1], gps_px[0]), rotate_angle)

        # print(np.where(np.all(map==[0,0,255], axis=-1)))
        print(gps_px)
        print(rotated_gps_px)
        # quit()
        # print(slope)
        # print(slope_px)

        # cv.circle(map, (np.int32(last_gps_px[0]), np.int32(last_gps_px[1])), 5, (0,255,0))
        # cv.imwrite('last_gps.png', rotated_map)
        traj_map = map_traj(rotated_map, radius, slope_px, pixel_meter, rotated_gps_px)
        # cv.circle(map, (np.int32(gps_px[0]), np.int32(gps_px[1])), 5, (0,255,0), thickness=-1)
        traj_map = skimage.transform.rotate(traj_map, -rotate_angle, resize=False, clip=True, preserve_range=True)
        cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/' + '3_2_osm/%s.png'%(x), traj_map)
        last_gps = gps
    # tile =get_tile(50.0377052, 9.2961994, 16)
    # IMU_pos(tile, 50.0377052, 9.2961994, 16)


    # print((xt,yt))
