#!/bin/bash

ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:/ros_ws"
. /opt/conda/etc/profile.d/conda.sh
conda activate vilbert-mt

# Build vqa maskrcnn, TODO should happen in Dockerfile (needs cuda)
cd /build/vqa-maskrcnn-benchmark || return
python setup.py build develop


# ROS
roscore &

echo "Waiting for ROS core to start..."
sleep 1

#rosrun object_segmentation_from_text model_inference_pipeline.py &
rosrun object_segmentation_from_text model_inference_vilbert.py &

rosrun object_segmentation_from_text image_buffer_from_path.py /ros_ws/object_segmentation_from_text/data/boy.jpg > /dev/null &

rosrun rviz rviz -d /ros_ws/rviz-cfg.rviz &

#echo 'boy' | rosrun object_segmentation_from_text caption_buffer_from_console.py > /dev/null &


cd /ros_ws || return
terminator -u