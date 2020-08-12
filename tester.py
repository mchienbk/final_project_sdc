import my_params
import csv
from interpolate_poses import interpolate_poses
import numpy as np
import numpy.matlib as ml
from transform import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import cv2

# test_file = './/groundtruth//test.csv'
test_file = my_params.poses_file

### Interpolate poses test
'''
# origin_timestamp = 1446206196878817
# pose_timestamps = 1446206198066328

with open(test_file) as vo_file:
    vo_reader = csv.reader(vo_file)
    headers = next(vo_file)

    vo_timestamps = [0]
    abs_poses = [ml.identity(4)]

    lower_timestamp = 1446206196878817
    upper_timestamp = 1446206198066328

    for row in vo_reader:
        timestamp = int(row[0])
        if timestamp < lower_timestamp:
            vo_timestamps[0] = timestamp
            continue

        vo_timestamps.append(timestamp)

        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)
        abs_pose = abs_poses[-1] * rel_pose
        abs_poses.append(abs_pose)
        if timestamp >= upper_timestamp:
            break

requested_timestamps = [vo_timestamps[11] + 3554]
# print(len(vo_timestamps),len(abs_poses),len(requested_timestamps))
test_pose = interpolate_poses(vo_timestamps, abs_poses, requested_timestamps, origin_timestamp)
print(test_pose)
print(abs_poses[11])
'''

### pose transform test
'''
H_new=np.identity(4)
C_origin=np.zeros((3,1))
R_origin=np.identity(3)

xyzrpy = [4.255068e-01,-1.712746e-03,-2.812839e-02,1.789666e-03,-1.110501e-03,-9.086471e-04]
tranf = build_se3_transform(xyzrpy)
print('tranf',tranf)

R_new = tranf[0:3,0:3]
T_new = tranf[0:3,3]
print('R_new',R_new)
print('T_new',T_new)

H_final= np.hstack((R_new,T_new))
H_final=np.vstack((H_final,[0,0,0,1]))
print('H_final',H_final)

x_old=H_new[0][3]
z_old=H_new[2][3]
print("old",(x_old,z_old))

H_new=H_new@H_final

print('H_new',H_new)
x_test=H_new[0,3]
z_test=H_new[2,3]
print("new",(x_test,z_test))
'''


### generate pose from vo test

H_new=np.identity(4)
C_origin=np.zeros((3,1))
R_origin=np.identity(3)
abs_poses = [ml.identity(4)]

x_old=(0,0)
z_old=(0,0)

fig = plt.figure()
gs = plt.GridSpec(2,3)

with open(test_file) as vo_file:
    vo_reader = csv.reader(vo_file)
    headers = next(vo_file)

    for row in vo_reader:
        timestamp = int(row[0])
        datetime = dt.utcfromtimestamp(timestamp/1000000)
        print(datetime)

        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)
        # print('rel pose',rel_pose)

        # abs_pose = abs_poses[-1] * rel_pose
        # abs_poses.append(abs_pose)

        H_new=H_new@rel_pose
        x_test=H_new[0,3]
        z_test=H_new[2,3]

        # print("old: ",(x_old,z_old))
        # print("new: ",(x_test,z_test))
        print((x_old,z_old), " : ",(x_test,z_test))

        x_old=x_test
        z_old=z_test

        plt.plot(x_test,-z_test,'o',color='blue')
        plt.pause(0.01)
        # plt.legend(['Built-In','Our Code'])
    plt.show()