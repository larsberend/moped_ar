import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

video_path = '../100GOPRO/kandidaten/'
file = '2_3'
cap = cv.VideoCapture(video_path + file + '.mp4')
my_dpi = 96
s = 1

ret, frame1 = cap.read()
frame1 = cv.imread('optflow_test1.png')
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
prvs = cv.resize(prvs, (int(prvs.shape[1]/s), int(prvs.shape[0]/s)))
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255
frame_nr_int = 0
while(1):
    ret, frame2 = cap.read()
    frame2 = cv.imread('optflow_test2.png')
    next1 = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    next = cv.resize(next1, (np.int32(next1.shape[1]/s), np.int32(next1.shape[0]/s)))

    tvl1 = cv.optflow.DualTVL1OpticalFlow_create()
    # tvl1.setLambda(0.015)
    # tvl1.setEpsilon(0.001)
    flow = tvl1.calc(prvs, next, flow=None)
    # flow = cv.calcOpticalFlowFarneback(prvs, next, flow=None, pyr_scale=0.5, levels=0, winsize=10, iterations=1, poly_n=5, poly_sigma=1.2, flags=0)

    print(flow.shape)
    print(np.unique(flow[:,:,0], return_counts=True)[0].shape)
    print(np.unique(flow[:,:,1], return_counts=True))

    # flsh =
    flow = cv.resize(flow, (int(flow.shape[1]/10), int(flow.shape[0]/10)), interpolation=cv.INTER_NEAREST )

    # print(flow.shape)
    # print(flow[:,:,0].shape)
    # print(flow[1].shape)

    # X = np.arange(0, np.int32(next1.shape[1]), s)
    # Y = np.arange(0, np.int32(next1.shape[0]), s)

    # crete array with number of breakpoints
    X = np.linspace(0, next1.shape[1], flow.shape[1], dtype=np.int32)
    Y = np.linspace(0, next1.shape[0], flow.shape[0], dtype=np.int32)


    # print((int(next1.shape[1]/s), int(next1.shape[0]/s)))
    # print((X.shape,Y.shape))
    # print(next.shape)
    # print(X.shape)
    # print(Y.shape)
    # print(flow.shape[:][:])
    # U = np.zeros()


    U = flow[:,:,0]
    V = flow[:,:,1]

    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111)
    q = ax.quiver(X, Y, U, V, color=(0.0,0.9,0.0,0.9))
    # ax.quiverkey(q, X=0.3, Y=1.1, U=1, label = 'asdf')
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis('off')
    plt.tight_layout()
    plt.imshow(frame2)
    # plt.savefig('../100GOPRO/kandidaten/%s_optflow_vec/%s.png'%(file, frame_nr_int), bbox_inches='tight')
    plt.savefig('../100GOPRO/kandidaten/%s_optflow_vec/test.png'%(file), bbox_inches='tight')
    plt.close()
    quit()
    '''
    for y in range(0, prvs.shape[0], s):
        for x in range(0, prvs.shape[1], s):
            fxy = flow[y,x]
            cv.line(prvs, (x, y), (int(x+fxy[1]), int(y+fxy[0])), (0, 255, 0) )
            cv.circle(prvs, (int(x+fxy[1]), int(y+fxy[0])), 1, (0, 255, 0))
    '''

    '''

    resize(GetImg, prvs, Size(GetImg.size().width/s, GetImg.size().height/s) );
     cvtColor(prvs, prvs, CV_BGR2GRAY);


    void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
     for(int y = 0; y < cflowmap.rows; y += step)
            for(int x = 0; x < cflowmap.cols; x += step)
            {
                const Point2f& fxy = flow.at< Point2f>(y, x);
                line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                     color);
                circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
            }
        }
    '''




    # mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])


    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    # retval = cv.imwrite('../100GOPRO/kandidaten/%s_optflow/%s.png'%(file, frame_nr_int), bgr)
    # cv.imshow('frame2', prvs)
    # print(retval)
    # print('../100GOPRO/kandidaten/%s_optflow/%s.png'%(file, frame_nr_int))
    # print(frame_nr_int)
    frame_nr_int += 1
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    # elif k == ord('s'):
    #     cv.imwrite('opticalfb.png',frame2)
    #     cv.imwrite('opticalhsv.png',bgr)
    prvs = next
    # print('reached end')
cap.release()
cv.destroyAllWindows()
print(frame_nr_int)
