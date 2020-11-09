# object_segmentation_from_text
Fork of https://github.com/gtziafas/obj_segm_from_text, extended with ViLBERT support

# usage
The [docker](docker) folder contains the files needed to run the project in a Docker.
This eliminates the need to have ROS and CUDA configured on your system. The only dependencies
are docker and NVIDIA Container Toolkit.

There are still some steps that need to be taken before the project will run:
- Be sure to edit paths inside files to your own static images, rosbags etc.
- The zsgnet model weights are missing, too big of a file, follow instructions on the paper's [git page](https://github.com/TheShadow29/zsgnet-pytorch), our module will take care on how to implement inference
- For vilbert: detectron.pth from https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth and
a (fine-tuned) vilbert model need to be placed in the [vilbert_data](ros/vilbert_data) folder, refer to the ViLBERT 
[git page](https://github.com/facebookresearch/vilbert-multi-task) for details
- Open a few terminals (with terminator)
- All or some of these steps are performed automatically when using the docker configuration, can be toggled by 
(un)commenting lines in [the entrypoint](docker/entrypoint/entrypoint.sh)
```
# startup master
$roscore

# startup model inference, choose one:
# zsgnet:
$rosrun object_segmentation_from_text model_inference_pipeline.py
# or vilbert:
$rosrun object_segmentation_from_text model_inference_vilbert.py

# ONLY ONE of the following: define your visual input src (static images, rosbag in a loop, real-time data from
# depth sensors e.g. Kinect)
$rosrun object_segmentation_from_text image_buffer_from_path.py file_path  
$rosbag play -l bag_file 
$roslaunch openni_launch openni_launch

# setup visualization
$rosrun rviz rviz 

# run this ONLY if you have visual input from RGB-D data
# setup 3d extraction modules
$rosrun object_segmentation_from_text 3d_image_buffer.py 
$roslaunch object_segmentation_from_text 3d_extraction.launch
$rosrun tf static_transform_publisher 0 0 0 0 0 0 map camera_rgb_optical_frame 100

# startup caption buffering
$rosrun object_segmentation_from_text caption_buffer_from_console.py
 ```