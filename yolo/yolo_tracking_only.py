from __future__ import division
import os
import sys
import cv2 
import time
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

# import argparse

sys.path.append('D:/Github/final_project_sdc')
import my_params

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

SAVE_VIDEO = 1
if __name__ == '__main__':
    # args = arg_parse()
    # confidence = float(args.confidence)
    # nms_thesh = float(args.nms_thresh) 
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

    # model.net_info["height"] = args.reso
    model.net_info["height"] = my_params.yolo_reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, CUDA), CUDA)
    model.eval()
    
    frames = 0
    start = time.time() 
    current_chunk = 0

    scale = 0.5
    width = int(1280 * scale)
    height = int(960 * scale)
    dim = (width, height)

    classes = load_classes(my_params.yolo_data + 'fix.names')
    colors = pkl.load(open(my_params.yolo_data + "pallete", "rb"))

    # Make save
    if (SAVE_VIDEO):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(my_params.output_dir + '\\yolo_detect.mp4', fourcc, 30, (1280, 960))

    timestamps_path = my_params.dataset_patch + 'stereo.timestamps'
    timestamps_file = open(timestamps_path)
    for line in timestamps_file:
        tokens = line.split()
        # datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
        # print(filename)
        chunk = int(tokens[1])
        filename = os.path.join(my_params.reprocess_image_dir + '\\', tokens[0] + '.png')

        if not os.path.isfile(filename):
            if chunk != current_chunk:
                print("Chunk " + str(chunk) + " not found")
                current_chunk = chunk
            continue

        current_chunk = chunk
        # print(filename)

        frame = cv2.imread(filename)
        # resize
        # frame= cv2.rectangle(frame,(np.float32(50),np.float32(np.shape(frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
        # frame = cv2.resize(frame,dim)
        # resize
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
        
        list(map(lambda x: write(x, orig_im), output))
        
        cv2.imshow("frame", orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))   
        
        # Save save
        if (SAVE_VIDEO):   
            video.write(orig_im)
        
    print('done!')
    cv2.destroyAllWindows()

