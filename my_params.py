

project_patch = 'D:\\Github\\final_project_sdc\\'
dataset_patch = 'D:\\Dataset\\20151030\\'

# project_patch = "D:\\Github\\final_project_sdc\\"
# dataset_patch = "D:\\Dataset\\20140514\\"


poses_file = dataset_patch + 'vo\\vo.csv'
extrinsics_dir = project_patch + 'extrinsics'
laser_dir = dataset_patch + 'ldmrs'

camera = 'stereo\\centre'
camera_name = 'stereo'

image_dir = dataset_patch + camera
model_dir = project_patch + 'models'

model = model_dir + camera_name

reprocess_image_dir = image_dir + '_processed'

yolo_cfg = project_patch + 'yolo\\cfg\\yolov3.cfg'
yolo_weights = project_patch + 'yolo\weights\yolov3.weights'
yolo_data = project_patch +'yolo\\data\\'