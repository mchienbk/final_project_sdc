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
from build_pointcloud import build_pointcloud
from itertools import islice

import my_params

from interpolate_poses import interpolate_poses
from transform import build_se3_transform, so3_to_quaternion
from camera_model import CameraModel


# Data location
vo_directory = my_params.dataset_patch + 'vo//vo.csv'
lidar_folder_path = my_params.laser_dir
lidar_timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'

output_dir = my_params.output_dir+'\\lidar_'+ my_params.dataset_no + '\\'
# output_points_patch = my_params.output_dir+'\\'+ my_params.dataset_no + '\\'#'_lidar_points.csv'

# Making Video
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# vw = cv2.VideoWriter('laser_scan.mp4', fourcc, 30, (600, 800))


# Intial data
# xyzrpy = [0, 0, 0, -0.090749, -0.000226, 4.211563]
# abs_poses = [abs_pose]


vo_on_map = []
obj_on_map = []
vo_timestamps = [0]
abs_poses = [ml.identity(4)]

scale = 1

abs_pose = build_se3_transform(my_params.xyzrpy)
abs_poses.append(abs_pose)
# Reading vo
print('Loading visual odometry...')
with open(vo_directory) as vo_file:
    vo_reader = csv.reader(vo_file)
    headers = next(vo_file)

    index = 0
    for row in vo_reader:
        timestamp = int(row[0])
        vo_timestamps.append(timestamp)
        # datetime = dt.utcfromtimestamp(timestamp/1000000)
       
        index += 1
        # if index > 3000 : 
        #     end_time = timestamp
        #     break
        # if index > 3000 : 

        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)
        abs_pose = abs_poses[-1] * rel_pose
        abs_poses.append(abs_pose)
end_time = timestamp
vo_file.close()

print('num_of_point:', len(abs_poses))
# print('len_abs_poses',(np.array(abs_poses)).shape)
# print('len_vo_timestamps',(np.array(vo_timestamps)).shape)


plt.figure()
abs_quaternions = np.zeros((4, len(abs_poses)))
abs_positions = np.zeros((3, len(abs_poses)))
for i, pose in enumerate(abs_poses):
    abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3])
    abs_positions[:, i] = np.ravel(pose[0:3, 3])
    point = (-abs_positions[0,i],abs_positions[1,i])        # -x, y of point
    plt.scatter(point[0],point[1],c='b',marker='.', zorder=1)
    vo_on_map.append(point)
# plt.show()
vo_on_map = np.array(vo_on_map)
vo_timestamps = np.array(vo_timestamps)

# # Read lidar extrnsics
with open(os.path.join(my_params.extrinsics_dir, 'ldmrs.txt')) as extrinsics_file:
    extrinsics = next(extrinsics_file)
G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
# print('G_posesource_laser',G_posesource_laser)

# Read pointcloud 
print('Matching lidar pointcloud...')
requested_timestamps = []
lidar_timestamps = []
with open(lidar_timestamps_path) as lidar_timestamps_file:
    num_of_laser = 0
    for _,row in enumerate(lidar_timestamps_file):
        lidar_timestamp = int(row.split(' ')[0])
        
        if (lidar_timestamp > end_time): 
            print("num_of_laser_scan: ",num_of_laser)
            break

        requested_timestamps.append(lidar_timestamp)
        lidar_timestamps.append(lidar_timestamp)
        num_of_laser += 1
lidar_timestamps_file.close()
# np.savetxt('requested_timestamps.csv', requested_timestamps, delimiter=",")

# Transform pointcloud
poses = interpolate_poses(vo_timestamps, abs_poses, requested_timestamps, vo_timestamps[0])
for i in range(num_of_laser):
    # if i > 500: break
    if lidar_timestamps[i] >= end_time: break

    # pose = poses[i]
    pose = np.dot(poses[i],G_posesource_laser)
    # print('pose',pose)
    # abs_quaternions = so3_to_quaternion(pose[0:3, 0:3])

    file_path = os.path.join(lidar_folder_path + '\\' + str(lidar_timestamps[i]) + '.bin')
    scan_file = open(file_path)
    objs_data = np.fromfile(scan_file, np.double)
    scan_file.close()

    xyz = objs_data.reshape((len(objs_data) // 3, 3)).transpose()
    xyz = np.array(xyz)
    xyz1 = np.vstack((xyz,np.ones(xyz.shape[1])))

    new_xyz1 = pose*xyz1
    x_obj = np.array(new_xyz1[0,:])
    y_obj = np.array(new_xyz1[1,:])
    plt.scatter(-x_obj,y_obj,c='r',marker='.', zorder=0)

    obj_on_map.append((new_xyz1[0:1,:]))

    # save_to_csv = []
    save_csv = np.vstack((x_obj,y_obj))
    np.savetxt((output_dir + str(lidar_timestamps[i]) + '.csv'), save_csv.transpose(), delimiter=",")

    
plt.show()



# file_path = os.path.join(lidar_folder_path + '\\' + str(lidar_timestamps) + '.bin')
# scan_file = open(file_path)
# objs_data = np.fromfile(scan_file, np.double)
# scan_file.close()
# objs_data = objs_data.reshape((len(objs_data) // 3, 3)).transpose()

# for i in range(objs_data.shape[1]):
#     xyz = objs_data[:,i]
#     new_xyz = abs_quaternions*xyz
#     x_obj=float(new_xyz[0])
#     y_obj=float(new_xyz[1])
#     obj_on_map.append((x_obj,y_obj))
# index = index + 1
# print("call: ",index)
# if (index > 100): break


# obj_map = np.array(obj_map)
# n_points = obj_map.shape[0]

# traj_min_x = np.min(traj_map[:, 0])
# traj_min_y = np.min(traj_map[:, 1])
# traj_max_x = np.max(traj_map[:, 0])
# traj_max_y = np.max(traj_map[:, 1])

# obj_min_x = np.min(obj_map[:, 0])
# obj_min_y = np.min(obj_map[:, 1])
# obj_max_x = np.max(obj_map[:, 0])
# obj_max_y = np.max(obj_map[:, 1])

# grid_min_x = scale*min(traj_min_x, obj_min_x)
# grid_min_y = scale*min(traj_min_y, obj_min_y)
# grid_max_x = scale*max(traj_max_x, obj_max_x)
# grid_max_y = scale*max(traj_max_y, obj_max_y)

# grid_res = [int(grid_max_x - grid_min_x), int(grid_max_y - grid_min_y)]

# print ('amplitude_x: ', grid_min_x, grid_max_x )
# print ('amplitude_y: ', grid_min_y, grid_max_y)
# print ('grid_res: ', grid_res)

# print(traj_map.shape)
# print(obj_map.shape)

# visit_counter = np.zeros(grid_res, dtype=np.int32)
# occupied_counter = np.zeros(grid_res, dtype=np.int32)

# grid_cell_size_x = (grid_max_x - grid_min_x) / float(grid_res[0])
# grid_cell_size_y = (grid_max_y - grid_min_y) / float(grid_res[1])

# norm_factor_x = float(grid_res[0] - 1) / float(grid_max_x - grid_min_x)
# norm_factor_y = float(grid_res[1] - 1) / float(grid_max_y - grid_min_y)
# print ('norm_factor_x: ', norm_factor_x)
# print ('norm_factor_z: ', norm_factor_y)


# for point_id in range(n_points):
#     point_x = int((scale*obj_map[point_id,0]- grid_min_x) * norm_factor_x)
#     point_y = int((scale*obj_map[point_id,1]- grid_min_y) * norm_factor_y)

#     visit_counter[point_x, point_y] = 255

# grid_map = np.zeros(grid_res, dtype=np.float32)
# grid_map_thresh = np.zeros(grid_res, dtype=np.uint8)

# for x_p in range(grid_res[0]):
#     for y_p in range(grid_res[1]):
#         if (visit_counter[x_p, y_p] == 0 ):
#             grid_map_thresh[x_p, y_p] = 0
#         if (visit_counter[x_p, y_p] > 0 ):
#             grid_map_thresh[x_p, y_p] = 255

# plt.xticks(size=12,color = "black")
# plt.yticks(size=12,color = "black")
# plt.imshow(grid_map_thresh,cmap="plasma")       # True = yellow
# plt.show()


# print("Done!!")




