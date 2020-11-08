#!/usr/bin/env python3

import cv2
import numpy as np
import roslib
from PIL import Image as PILImage

from vilbert_grounding import VILBertGrounding

roslib.load_manifest('object_segmentation_from_text')
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage

IMAGE_SIZE = 300


class InferenceNode:
    def __init__(self, vilbert_grounding_model: VILBertGrounding):
        self.vilbert_model = vilbert_grounding_model
        rospy.init_node('model_inference_vilbert', anonymous=True)
        self.rgb_pub = rospy.Publisher('/object_segmentation_from_text/RGB_with_box', ROSImage, queue_size=10)
        self.caption_sub = rospy.Subscriber("/caption_buffer_from_console/caption", String, self.handle_caption)
        self.rgb_sub_static = rospy.Subscriber("/image_buffer_from_path/RGB", ROSImage, self.handle_image)
        self.rgb_sub_camera = rospy.Subscriber("/camera/rgb/image_rect_color", ROSImage, self.handle_image)

        self.img_msg = None
        self.query = None

        print('Ready to receive input')

    def handle_caption(self, caption: String):
        self.query = caption.data

    def handle_image(self, img_msg: ROSImage):
        self.img_msg = img_msg
        self.inference(img_msg)

    def inference(self, img: ROSImage):
        if not self.query:
            return

        h, w = img.height, img.width
        cv2_image = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w, -1)
        pil_img = PILImage.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

        bbox = self.vilbert_model.evaluate(pil_img, self.query)

        if bbox is None:
            failed_img = cv2.putText(cv2_image, f'"{self.query}" not found', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                     (0, 0, 0xFF), thickness=2)
            self.publish_image(failed_img.tobytes())
            return

        image = draw_box(cv2_image, self.query, bbox)

        self.publish_image(image.tobytes())

    def publish_image(self, data: bytes):
        self.img_msg.data = data

        self.rgb_pub.publish(self.img_msg)


def draw_box(image, label, bbox):
    with_bbox = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0xFF), 2)
    labeled = cv2.putText(with_bbox, label, (bbox[0] + 2, bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0xFF),
                          thickness=2)

    return labeled


if __name__ == '__main__':
    print('Starting...')
    model = VILBertGrounding()
    print('Starting node...')
    InferenceNode(model)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
