# Primary patch
dataset_no = '20151030'
project_patch = 'D:\\Github\\final_project_sdc\\'
dataset_patch = 'D:\\Dataset\\' + dataset_no + '\\'
# dataset_patch = "D:\\Dataset\\20140514\\"

# camera
camera = 'stereo\\centre'
camera_name = 'stereo'

# Folder directory
image_dir = dataset_patch + camera
laser_dir = dataset_patch + 'ldmrs'
model_dir = project_patch + 'models'
extrinsics_dir = project_patch + 'extrinsics'
reprocess_image_dir = image_dir + '_processed'
output_dir = project_patch + 'output'


# Data file directory
model = model_dir + camera_name
poses_file = dataset_patch + 'vo\\vo.csv'

# YOLO v3 Video Detection Module
yolo_video = "video.avi"
yolo_dataset = "pascal"
yolo_confidence = 0.5   # default = 0.5 - "Do tin cay" class of output
yolo_nms_thresh = 0.4   # default = 0.4
yolo_cfg = project_patch + 'yolo\\cfg\\yolov3.cfg'
yolo_weights = project_patch + 'yolo\weights\yolov3.weights'
yolo_data = project_patch +'yolo\\data\\'
yolo_reso = 416          # default = "416"

yolo_test_img = project_patch + 'yolo\\test_img\\hinh-2.jpg'
