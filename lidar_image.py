import os
import re
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
# from config import *

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

import my_params

if __name__ == "__main__":
   
    # Making save
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw = cv2.VideoWriter('laserrange_image.mp4', fourcc, 30, (1280, 960))

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', my_params.laser_dir).group(0)
    lidar_timestamps_path = os.path.join(my_params.dataset_patch, lidar + '.timestamps')

    camera = re.search('(stereo|mono_(left|right|rear))', my_params.image_dir).group(0)
    camera_timestamps_path = os.path.join(os.path.join(my_params.dataset_patch, camera + '.timestamps'))

    model = CameraModel(my_params.model_dir, my_params.image_dir)
    poses_file = my_params.poses_file
    extrinsics_dir  =  my_params.extrinsics_dir

    with open(os.path.join(os.path.join(my_params.extrinsics_dir, camera + '.txt'))) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

    timestamp = 0
    start_time = 0
    end_time = 0

    set_lidar_flg = 0
    lidar_timestamps_old = 0

    # Create a color
    color = [255,0,0]

    with open(camera_timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            
            if (i < 40):
                continue
            if (i > 45):
                break

            image_timestamp = int(line.split(' ')[0])
            image_path = os.path.join(my_params.image_dir, str(image_timestamp) + '.png')
            image = load_image(image_path, model)
            
            with open(lidar_timestamps_path) as lidar_timestamps_file:
                for j, row in enumerate(lidar_timestamps_file):
                    # print("j = ", j)    # Test
                    lidar_timestamps = int(row.split(' ')[0])
                    # print(lidar_timestamps)

                    if (lidar_timestamps > image_timestamp):
                        # set_lidar_flg = 1
                        start_time = image_timestamp
                        end_time = image_timestamp + 5e6
                        break

            lidar_timestamps_file.close()

            pointcloud, reflectance = build_pointcloud(my_params.laser_dir, poses_file, extrinsics_dir, 
                                                        start_time, end_time, lidar_timestamps)
            pointcloud = np.dot(G_camera_posesource, pointcloud)
                        
            uv, depth = model.project(pointcloud, image.shape)

            img2_path = os.path.join(my_params.image_dir + '//' + str(image_timestamp) + '.png')
            img2 = cv2.imread(img2_path)   # must processed imagte
            # img2 = load_image(img2_path,model)
            img1 = np.zeros_like(img2)


            for k in range(uv.shape[1]):
                x_p = (int)(np.ravel(uv[:,k])[0])
                y_p = (int)(np.ravel(uv[:,k])[1])

                color = (int(255-8*depth[k]),255-3*depth[k],50+3*depth[k])
                img1= cv2.circle(img1, (x_p, y_p), 1, color, 1) 
                img2= cv2.circle(img2, (x_p, y_p), 1, color, 1)             
           
            cv2.imwrite("pointcloud_" + str(i) + ".jpg", img1)
            cv2.imwrite("image_" + str(i) + ".jpg", img2)
 
            # key = cv2.waitKey(1)
            # if key & 0xFF == ord('q'):
            #     timestamps_file.close()
            #     print('stop!')
            #     cv2.destroyAllWindows()               
            #     break

    timestamps_file.close()
    print('done!')
    cv2.destroyAllWindows()