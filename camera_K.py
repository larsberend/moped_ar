import numpy as np

f = 0.00164104
u0 = 960
v0 = 540

pu = np.float64(0.00617 / 1920)
pv = np.float64(0.00455 / 1080)

pixel_mat = np.array([[1/pu, 0, u0],
                      [0, 1/pv, v0],
                      [0, 0, 1]], dtype=np.float64
                    )
focal_mat = np.array([[f,0,0],
                      [0,f,0],
                      [0,0,1]], dtype=np.float64
                    )

K = np.dot(pixel_mat, focal_mat)


K_homog = np.zeros((3,4))
K_homog[:, :-1] = K
# print(focal_mat)
# print(pixel_mat)
print(K)
print(K_homog)
