import os

DATA_PATH = '/ros_ws/vilbert_data'
path_join = os.path.join

DETECTRON_CONFIG = path_join(DATA_PATH, 'detectron_config.yaml')
DETECTRON_MODEL = path_join(DATA_PATH, 'detectron_model.pth')

PRETRAINED_BERT = 'bert-base-uncased'
BERT_CONFIG = path_join(DATA_PATH, 'bert_base_6layer_6conect.json')
VILBERT_MODEL = path_join(DATA_PATH, 'finetune_18_epoch4.bin')

TASK = 18
