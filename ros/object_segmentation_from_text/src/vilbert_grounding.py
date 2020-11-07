from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL.Image import Image
from pytorch_transformers import BertTokenizer
from script.extract_features import FeatureExtractor as BaseFeatureExtractor
from vilbert.vilbert import VILBertForVLTasks, BertConfig

import vilbert_cfg


class VILBertGrounding:
    """
    Visual grounding using ViLBERT. Adapted from demo.ipynb in ViLBERT repository.
    """

    def __init__(self):
        print('Initializing feature extractor')
        self.feature_extractor = FeatureExtractor()

        print('Initializing ViLBERT')
        self.vilbert = init_vilbert()

        print('Initializing tokenizer')
        self.tokenizer = init_tokenizer()

        self.task = [vilbert_cfg.TASK]

    def _prediction(self,
                    img,
                    question,
                    features,
                    spatials,
                    segment_ids,
                    input_mask,
                    image_mask,
                    co_attention_mask,
                    task_ids,
                    ):
        (vil_prediction,
         vil_prediction_gqa,
         vil_logit,
         vil_binary_prediction,
         vil_tri_prediction,
         vision_prediction,
         vision_logit,
         linguisic_prediction,
         linguisic_logit,
         attn_data_list) = self.vilbert(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_ids,
            output_all_attention_masks=True
        )

        height, width = img.shape[0], img.shape[1]

        # grounding:
        grounding_val, grounding_idx = torch.sort(vision_logit.view(-1), 0, True)

        output_idx = 0
        confidence = grounding_val[output_idx].item()
        print('Confidence:', confidence)

        idx = grounding_idx[output_idx]
        box = spatials[0][idx][:4].tolist()

        y1 = int(box[1] * height)
        y2 = int(box[3] * height)
        x1 = int(box[0] * width)
        x2 = int(box[2] * width)

        return [x1, y1, x2, y2]

    def custom_prediction(self, img, query, features, infos):
        tokens = self.tokenizer.encode(query)
        tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        max_length = 37
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [0] * (max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        text = torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0)
        input_mask = torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0)
        segment_ids = torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0)
        task = torch.from_numpy(np.array(self.task)).cuda().unsqueeze(0)

        num_image = len(infos)
        assert num_image > 0

        feature_list = []
        image_location_list = []
        image_mask_list = []
        for i in range(num_image):
            image_w = infos[i]['image_width']
            image_h = infos[i]['image_height']
            feature = features[i]
            num_boxes = feature.shape[0]

            g_feat = torch.sum(feature, dim=0) / num_boxes
            num_boxes = num_boxes + 1
            feature = torch.cat([g_feat.view(1, -1), feature], dim=0)
            boxes = infos[i]['bbox']
            image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            image_location[:, :4] = boxes
            image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                    image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))
            image_location[:, 0] = image_location[:, 0] / float(image_w)
            image_location[:, 1] = image_location[:, 1] / float(image_h)
            image_location[:, 2] = image_location[:, 2] / float(image_w)
            image_location[:, 3] = image_location[:, 3] / float(image_h)
            g_location = np.array([0, 0, 1, 1, 1])
            image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
            image_mask = [1] * (int(num_boxes))

            feature_list.append(feature)
            image_location_list.append(torch.tensor(image_location))
            image_mask_list.append(torch.tensor(image_mask))

        features = torch.stack(feature_list, dim=0).float().cuda()
        spatials = torch.stack(image_location_list, dim=0).float().cuda()
        image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()
        co_attention_mask = torch.zeros((num_image, num_boxes, max_length)).cuda()

        return self._prediction(img, text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask,
                                task)

    def evaluate(self, img: Image, query: str):
        features, infos = self.feature_extractor.get_detectron_features([img])

        img_tensor = torch.tensor(np.array(img))
        return self.custom_prediction(img_tensor, query, features, infos)


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.args = self.get_parser()
        self.detection_model = self._build_detection_model()

    def get_parser(self):
        return SimpleNamespace(model_file=vilbert_cfg.DETECTRON_MODEL,
                               config_file=vilbert_cfg.DETECTRON_CONFIG,
                               batch_size=1,
                               num_features=100,
                               feature_name="fc6",
                               confidence_threshold=0,
                               background=False,
                               partition=0)

    def _image_transform(self, img):
        """
        Copied from superclass, modified to accept PIL image instead of path
        """
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info


def init_vilbert() -> VILBertForVLTasks:
    config = BertConfig.from_json_file(vilbert_cfg.BERT_CONFIG)
    device = torch.device("cuda")

    model = VILBertForVLTasks.from_pretrained(
        vilbert_cfg.VILBERT_MODEL,
        config=config,
        num_labels=1,
        default_gpu=True,
    )

    model.to(device)
    model.eval()

    return model


def init_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained(
        vilbert_cfg.PRETRAINED_BERT, do_lower_case=True,
    )
