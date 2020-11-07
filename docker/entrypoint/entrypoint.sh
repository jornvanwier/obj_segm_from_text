#!/bin/bash

ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:/ros_ws"
. /opt/conda/etc/profile.d/conda.sh
conda activate vilbert-mt

#rosdep install --from-paths /ros_ws/object_segmentation_from_text --ignore-src -r -y

# ROS
roscore &

echo "Waiting for ROS core to start..."
sleep 1

rosrun object_segmentation_from_text model_inference_pipeline.py &

rosrun object_segmentation_from_text image_buffer_from_path.py /ros_ws/object_segmentation_from_text/data/boy.jpg &

rosrun rviz rviz &



cd /ros_ws || return
terminator -u