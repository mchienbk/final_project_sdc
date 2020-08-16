from __future__ import division
import os
import sys
import cv2 
import time
import re
import random 
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle as pkl
from torch.autograd import Variable
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
from datetime import datetime as dt


sys.path.append('D:/Github/final_project_sdc')
import my_params
from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

def get_test_input(input_dim, CUDA):
    img = cv2.imread('yolo/test_img/car.jpg')
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())    
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])  
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


if __name__ == '__main__':

    output_lidar_patch = my_params.output_dir+'\\lidar\\'
    output_yolo_patch = my_params.output_dir+'\\yolo\\'
    # Yolo intial 
    confidence = float(my_params.yolo_confidence)
    nms_thesh = float(my_params.yolo_nms_thresh)

    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(my_params.yolo_cfg)
    model.load_weights(my_params.yolo_weights)
    print("Network successfully loaded")

    model.net_info["height"] = my_params.yolo_reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, CUDA), CUDA)
    model.eval()

    # Lidar intial and camera pose
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', my_params.laser_dir).group(0)
    lidar_timestamps_path = os.path.join(my_params.dataset_patch, lidar + '.timestamps')

    camera = re.search('(stereo|mono_(left|right|rear))', my_params.image_dir).group(0)
    camera_timestamps_path = os.path.join(os.path.join(my_params.dataset_patch, camera + '.timestamps'))

    prj_model = CameraModel(my_params.model_dir, my_params.image_dir)
    poses_file = my_params.poses_file
    extrinsics_dir  =  my_params.extrinsics_dir

    with open(os.path.join(os.path.join(my_params.extrinsics_dir, camera + '.txt'))) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

    # Get image form timestamp
    with open(camera_timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            if (i < 100):
                continue
            image_timestamp = int(line.split(' ')[0])
            image_path = os.path.join(my_params.image_dir, str(image_timestamp) + '.png')
            image = load_image(image_path, prj_model)
            
            print('Image available')
            # Find correct Lidar data
            with open(lidar_timestamps_path) as lidar_timestamps_file:
                for j, row in enumerate(lidar_timestamps_file):
                    lidar_timestamps = int(row.split(' ')[0])

                    if (lidar_timestamps > image_timestamp):
                        start_time = image_timestamp
                        end_time = image_timestamp + 2e6    # default 5e6
                        print('Lidar available')
                        break
            # break

            lidar_timestamps_file.close()

            pointcloud, reflectance = build_pointcloud(my_params.laser_dir, poses_file, extrinsics_dir, 
                                                                start_time, end_time, lidar_timestamps)
            pointcloud = np.dot(G_camera_posesource, pointcloud)               
            uv, depth = prj_model.project(pointcloud, image.shape)

            # print('Now process with image')

            # Set input image
            frame_patch = os.path.join(my_params.reprocess_image_dir + '//' + str(image_timestamp) + '.png')
            frame = cv2.imread(frame_patch)   # must processed imagte
    
            # scale = 0.5
            width, height = frame.shape[1], frame.shape[0]
            # dim = (int(scale*width), int(scale*height))
    
            pframe = frame.copy()
            pcloud = np.zeros((height,width),dtype=float)

            # Make lidar depth map
            for k in range(uv.shape[1]):
                if (depth[k]>50): continue
                x_lidar = (int)(np.ravel(uv[:,k])[0])
                y_lidar = (int)(np.ravel(uv[:,k])[1])
                pcloud[y_lidar,x_lidar] = float(depth[k])

            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)

            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
            
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            classes = load_classes(my_params.yolo_data + 'coco.names')
            colors = pkl.load(open(my_params.yolo_data + "pallete", "rb"))
   
            # Prediction origin image only
            # list(map(lambda x: write(x, orig_im), output))   
            # cv2.imshow("frame", orig_im)

            # Prediction image with pointcloud
            for k in range(uv.shape[1]):
                x_lidar = (int)(np.ravel(uv[:,k])[0])
                y_lidar = (int)(np.ravel(uv[:,k])[1])
                color = (int(255-8*depth[k]),255-3*depth[k],50+3*depth[k])
                pframe= cv2.circle(pframe, (x_lidar, y_lidar), 1, color, 1) 
            
            list(map(lambda x: write(x, pframe), output))   
            cv2.imshow("output", pframe)

            # Output bbox
            # print('Number of bbox',output.shape[0])
            bframe = frame.copy()

            # Draw distance on image
            # list(map(lambda x: print(x), output))
            for i in range(int(output.shape[0])):
                # print(output[i,:])
                x = output[i,:]
                c1 = tuple(x[1:3].int())    
                c2 = tuple(x[3:5].int())
                cv2.rectangle(bframe, c1, c2, (255,0,0), 1)
                cls = int(x[-1])  
                label = "{0}".format(classes[cls])
                cv2.putText(bframe, label, (c1[0], c2[1]), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,255], 2)
                print(label)

                rec = pcloud[c1[1]:c2[1],c1[0]:c2[0]] #(y,x)
                dis = '{:06.2f}'.format(10*np.mean(rec))
                print(dis)
                cv2.putText(bframe, str(dis)+' m', (c1[0], c2[1]-30), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,255], 2)

            cv2.imshow("bframe", bframe)

            # Save lidar np matrix
            np.savetxt(output_lidar_patch + str(image_timestamp) +'.csv', pcloud, delimiter=",")
            # Save output yolo
            np.savetxt(output_yolo_patch + str(image_timestamp) +'.csv', output, delimiter=",")

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'): break

    timestamps_file.close()
    print('done!')
    cv2.destroyAllWindows()


    

