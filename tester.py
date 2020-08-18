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
def interpolate_poses():

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


### pose transform test
def pose_transform():

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



### generate pose from vo test
def pose_from_vo():
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


# ### DRAW IMAGE CV2
def draw_test():
    img = np.zeros((600,800,3), np.uint8)
    img= cv2.rectangle(img,(np.float32(50),np.float32(100)),(np.float32(100),np.float32(200)),(255,0,0),-1)

    # ray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imshow('image', img)

    cv2.waitKey(0)


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
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

if __name__ == "__main__":
    # import torch 
    # import torch.nn as nn
    # from torch.autograd import Variable
    # print("Start")
    # img = cv2.imread('D:\\Github\\final_project_sdc\\yolo\\test_img\\bear.jpg')
    # img, orig_im, dim = prep_image(img, 300)
    
    import random
    data = [[1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 1]]
    map = np.zeros((100,100),dtype=bool)
    i = 0; j = 0
    for i in range(100):
        for j in range(100):
            map[i,j] = random.randint(a=0,b=1)
    print(map[:10,:10])

    # random.randint(a=0,b=1)
    plt.xticks(size=12,color = "black")
    plt.yticks(size=12,color = "black")
    # plt.figure(figsize=(10,10))
    plt.imshow(map,cmap="plasma")       # True = yellow
    plt.show()
