


import os
import cv2
import csv
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy.matlib as ml

import my_params

from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from transform import build_se3_transform
from camera_model import CameraModel

# Load data

vo_directory = my_params.dataset_patch + 'vo//vo.csv'

C_origin=np.zeros((3,1))
R_origin=np.identity(3)
# xyzrpy = [0, 0, 0, -0.090749, -0.000226, 4.211563]

xyzrpy = [0, 0, 0, -0.083514, -0.009355, 1.087759]

abs_pose = build_se3_transform(xyzrpy)
abs_poses = [abs_pose]
H_new = abs_pose
print(H_new)
# fix start point
x_old=(0,0)
z_old=(0,0)

fig = plt.figure()
gs = plt.GridSpec(2,3)
vo_timestamps = []
with open(vo_directory) as vo_file:
    vo_reader = csv.reader(vo_file)
    headers = next(vo_file)

    index = 0
    for row in vo_reader:
        timestamp = int(row[0])
        vo_timestamps.append(timestamp)
        datetime = dt.utcfromtimestamp(timestamp/1000000)
        # print(datetime)

        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)
        # print('rel pose',rel_pose)

        # abs_pose = abs_poses[-1] * rel_pose
        # abs_poses.append(abs_pose)

        H_new=H_new@rel_pose
        x_test=H_new[1,3]
        z_test=H_new[0,3]

        # print("old: ",(x_old,z_old))
        # print("new: ",(x_test,z_test))
        # print((x_old,z_old), " : ",(x_test,z_test))

        x_old=x_test
        z_old=z_test
        if (index<500):
            plt.plot(x_test, z_test,'.',color='red')
        else:
            plt.plot(x_test, z_test,'.',color='blue')
        index = index + 1

        abs_poses.append(H_new)
    plt.pause(0.01)

# # plt.show()
# vo_timestamps = vo_timestamps[1:]
# abs_poses = abs_poses[1:]
# print('vo_timestamps',len(vo_timestamps))   #2892
# print('abs_poses',len(abs_poses[1:]))       #2892

lidar_timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'
lidar_folder_path = my_params.laser_dir

# timestamp = 0
#     with open(timestamps_path) as timestamps_file:
#         for i, line in enumerate(timestamps_file):
#             # if (i > 100): break

#             timestamp = int(line.split(' ')[0])

#             datetime = dt.utcfromtimestamp(timestamp/1000000)
#             print(datetime)

#             file_path = os.path.join(lidar_folder_path + '\\' + str(timestamp) + '.bin')
#             scan_file = open(file_path)

#             data = np.fromfile(scan_file, np.double)
#             scan_file.close()
#             data = data.reshape((len(data) // 3, 3)).transpose()

#             # Create a color
#             color = [255,0,0]

#             # Blank image
#             img = np.zeros((600,800,3),np.uint8)

#             # Change one by one pixel
#             for i in range(data.shape[1]):
#                 x = 5*data[0][i]
#                 y = 5*data[1][i]
#                 if (x > 599) : x = 599
#                 if (x < 1) : x = 1
#                 if (y < -400) : y = -399 
#                 if (y > 400) : y = 399 
#                 img[600-(int(x)),(int(y)) + 400]=color

#             # Save
#             cv2.imshow("result",img)
#             vw.write(img)

#             # Press 'q' to exit!
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#     print('done!')
#     cv2.destroyAllWindows()







