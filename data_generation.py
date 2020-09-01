import os
import cv2
import re
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
import my_params



def generate_pointcloud():

    WRITE_IMAGE = 1
    DRAW_MAP = 1

    # Data input
    image_dir = my_params.image_dir
    ldrms_dir = my_params.laser_dir
    lmsfront_dir = my_params.lmsfront_dir
    lmsrear_dir  = my_params.lmsrear_dir
    poses_file = my_params.poses_file
    models_dir = my_params.model_dir
    extrinsics_dir = my_params.extrinsics_dir

    output_img_dir   = my_params.output_dir2 + 'pl_img_'   + my_params.dataset_no + '//'
    output_ldrms_dir = my_params.output_dir2 + 'pl_ldrms_' + my_params.dataset_no + '//'
    output_front_dir = my_params.output_dir2 + 'pl_front_' + my_params.dataset_no + '//'
    output_rear_dir  = my_params.output_dir2 + 'pl_rear_'  + my_params.dataset_no + '//'
    
    map_ldrms_dir = my_params.output_dir2 + 'map_ldrms_' + my_params.dataset_no + '//'
    map_front_dir = my_params.output_dir2 + 'map_front_' + my_params.dataset_no + '//'
    map_rear_dir  = my_params.output_dir2 + 'map_rear_'  + my_params.dataset_no + '//'


    model = CameraModel(models_dir, image_dir)
    extrinsics_path = os.path.join(extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle

    image_size = (960,1280,3)

    timestamps_path = os.path.join(my_params.dataset_patch + model.camera + '.timestamps')
    timestamp = 0
    plt.figure()
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            
            if i < 799:
                # print('open image index', i)
                # timestamp = int(line.split(' ')[0])
                # break
                continue

            timestamp = int(line.split(' ')[0])
            start_time = timestamp - 1e6
            
            frame_path = os.path.join(my_params.reprocess_image_dir + '//' + str(timestamp) + '.png')
            frame = cv2.imread(frame_path) 
            
            print('process image ',i,'-',timestamp)

            if i < 4: 
                if(WRITE_IMAGE):
                    savefile_name = output_dir + '\\lms_front_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png'
                    cv2.imwrite(savefile_name, frame)           
                continue

            pl_ldrms = np.zeros((960,1280),dtype=int)
            pl_front = np.zeros((960,1280),dtype=int)

            

            # LDRMS
            ldrms_pointcloud, _ = build_pointcloud(ldrms_dir, poses_file, extrinsics_dir,
                                            start_time, timestamp + 2e6, timestamp)
            ldrms_pointcloud = np.dot(G_camera_posesource, ldrms_pointcloud)
            uv, depth = model.project(ldrms_pointcloud,image_size)

            x_p = np.ravel(uv[0,:])
            y_p = np.ravel(uv[1,:])
            z_p = np.ravel(depth)

            map_ldrms = (np.array(ldrms_pointcloud[0:3,:])).transpose()
            map_ldrms_filename = map_ldrms_dir + str(timestamp) + '.csv'
            np.savetxt(map_ldrms_filename, map_ldrms, delimiter=",")

            if (DRAW_MAP):
                map_x = [numeric_map_x for numeric_map_x in np.array(ldrms_pointcloud[0,:])]
                map_y = [numeric_map_y for numeric_map_y in np.array(ldrms_pointcloud[1,:])]
                map_z = np.array(ldrms_pointcloud[2,:])

                plt.scatter((map_y),(map_x),c='b',marker='.', zorder=1)

            # LDRMS pointcloud to CSV
            # for k in range(uv.shape[1]):

            #     pl_ldrms[int(y_p[k]),int(x_p[k])] = int(100*depth[k])

            # ldrms_filename = output_ldrms_dir + str(timestamp) + '.csv'
            # np.savetxt(ldrms_filename, pl_ldrms, delimiter=",")
        

            # LMS-FRONT
            front_pointcloud, _ = build_pointcloud(lmsfront_dir, poses_file, extrinsics_dir,
                                            start_time, timestamp + 1e6, timestamp)
            front_pointcloud = np.dot(G_camera_posesource, front_pointcloud)
            wh,xrange = model.project(front_pointcloud,image_size)

            x_f = np.ravel(wh[0,:])
            y_f = np.ravel(wh[1,:])
            z_f = np.ravel(xrange)

            map_front = (np.array(front_pointcloud[0:3,:])).transpose()
            map_front_filename = map_front_dir + str(timestamp) + '.csv'
            np.savetxt(map_front_filename, map_front, delimiter=",")

            if (DRAW_MAP):
                map_x = [numeric_map_x for numeric_map_x in np.array(front_pointcloud[0,:])]
                map_y = [numeric_map_y for numeric_map_y in np.array(front_pointcloud[1,:])]
                map_z = [-numeric_map_z for numeric_map_z in np.array(front_pointcloud[2,:])]

                plt.scatter((map_y),(map_z),c='r',marker='.', zorder=1)

            # LMS-FRONT pointcloud to CSV
            # for k in range(wh.shape[1]):

            #     pl_ldrms[int(y_f[k]),int(x_f[k])] = int(100*xrange[k])

            # front_filename = output_front_dir + str(timestamp) + '.csv'
            # np.savetxt(front_filename, pl_ldrms, delimiter=",")

            # LMS-REAR

            rear_pointcloud, _ = build_pointcloud(lmsrear_dir, poses_file, extrinsics_dir,
                                            start_time, timestamp + 2e6, timestamp)
            rear_pointcloud = np.dot(G_camera_posesource, rear_pointcloud)

            map_rear = (np.array(rear_pointcloud[0:3,:])).transpose()
            map_rear_filename = map_rear_dir + str(timestamp) + '.csv'
            np.savetxt(map_rear_filename, map_rear, delimiter=",")

            if (DRAW_MAP):
                map_x = [numeric_map_x for numeric_map_x in np.array(rear_pointcloud[0,:])]
                map_y = [-numeric_map_y for numeric_map_y in np.array(rear_pointcloud[1,:])]
                map_z = [numeric_map_z for numeric_map_z in np.array(rear_pointcloud[2,:])]

                plt.scatter((map_y),(map_z),c='g',marker='.', zorder=1)


            if (WRITE_IMAGE):
                for k in range(uv.shape[1]):

                    color = (int(255-8*depth[k]),255-3*depth[k],50+3*depth[k])
                    frame= cv2.circle(frame, (int(x_p[k]), int(y_p[k])), 1, color, 1)

                for k in range(wh.shape[1]):

                    color = (int(255-8*xrange[k]),255-3*xrange[k],50+3*xrange[k])
                    frame= cv2.circle(frame, (int(x_f[k]), int(y_f[k])), 1, color, 1)

                cv2.imshow('frame',frame)
                image_filename = output_img_dir   + str(timestamp) + '.png'
                cv2.imwrite(image_filename, frame)
                cv2.waitKey(1)
            # plt.show()
            plt.pause(0.1)
            # TEST
            # print('_')
            # data = np.loadtxt(ldrms_filename, delimiter=',')
            # cv2.imshow('image',data)
            # cv2.waitKey(0)

if __name__ == "__main__":

    generate_pointcloud()


    cv2.destroyAllWindows()
    print('Done!')
