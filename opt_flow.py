import cv2 as cv
import numpy as np

video_path = '../100GOPRO/kandidaten/'
file = '2_3'
cap = cv.VideoCapture(video_path + file + '.mp4')

s = 5

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
prvs = cv.resize(prvs, (int(prvs.shape[1]/s), int(prvs.shape[0]/s)))
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255
frame_nr_int = 0
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    # next = cv.resize(prvs, (int(next.shape[1]/s), int(next.shape[0]/s)))

    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    for y in range(0, prvs.shape[0], s):
        for x in range(0, prvs.shape[1], s):
            fxy = flow[y,x]
            cv.line(prvs, (x, y), (int(x+fxy[1]), int(y+fxy[0])), (0, 255, 0) )
            cv.circle(prvs, (int(x+fxy[1]), int(y+fxy[0])), 1, (0, 255, 0))


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
    cv.imshow('frame2', prvs)
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
    print('reached end')
cap.release()
cv.destroyAllWindows()
print(frame_nr_int)
