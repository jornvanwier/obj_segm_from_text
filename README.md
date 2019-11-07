# object_segmentation_from_text
Code for the final project of the RUG's course "Cognitive Robotics" (November 2019), based on: https://github.com/TheShadow29/zsgnet-pytorch

# demo
https://www.youtube.com/watch?v=kgQgaghf71o

# usage
- Be sure to edit paths inside files to your own static images, rosbags etc.
- The model weights are missing, too big of a file, follow instructions on the above git page to train the ZSG network
- Open 5 terminals:
```
  $roscore
  $rosrun oject_segmentation_from_text image_buffer_from_path.py (for static image), or rosbag play for recorded data)
  $rosrun object_segmentation_from_text modeL_inference_pipeline.py
  $rosrun object_segmentation_from_text caption_buffer_from_console.py
  $rosrun rviz rviz 
```

# todo
- Use depth_image_proc package to segment also PointCloud2 of the captioned object from RGBD data
- Implement a speech2text module for online speech captioning instead of console input
