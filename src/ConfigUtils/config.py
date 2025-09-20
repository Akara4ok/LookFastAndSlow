import yaml
import os
import logging
import ConfigUtils.config_const as const

class Config:
    def __init__(self, path: str):
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir, path)
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)['ObjectDetector']
            self.config = self.set_defaults_to_config(self.config)
            
    def get_dict(self):
        return self.config
            
    def set_defaults_to_config(self, config: dict) -> dict:
        config = Config.parse_config_key(config, 'seed', const.SEED)
        #data
        config = Config.parse_double_key(config, 'data', 'path', const.DATASET_PATH)
        config = Config.parse_double_key(config, 'data', 'test_percent', const.TEST_PERCENT)
        
        #train
        config = Config.parse_double_key(config, 'train', 'epochs', const.EPOCHS)
        config = Config.parse_double_key(config, 'train', 'tensorboard_path', const.LOGS_PATH)
        os.makedirs(config['train']['tensorboard_path'], exist_ok=True)
        config = Config.parse_double_key(config, 'train', 'batch_size', const.BATCH_SIZE)
        config = Config.parse_double_key(config, 'train', 'augmentation', const.AUGMENTATION)
        
        #model
        config = Config.parse_double_key(config, 'model', 'path', const.MODEL_PATH)
        os.makedirs(os.path.dirname(config['model']['path']), exist_ok=True)
        config = Config.parse_double_key(config, 'model', 'img_size', const.IMG_SIZE)
        config = Config.parse_double_key(config, 'model', 'fast_width', const.FAST_WIDTH)
        config = Config.parse_double_key(config, 'model', 'slow_width', const.SLOW_WIDTH)
        
        #lr
        config = Config.parse_double_key(config, 'lr', 'initial_lr', const.LR)
        config = Config.parse_double_key(config, 'lr', 'min_lr', const.MIN_LR)
        
        #anchors
        config = Config.parse_double_key(config, 'anchors', 'variances', const.VARIANCES)
        config = Config.parse_double_key(config, 'anchors', 'iou_threshold', const.IOU_THRESHOLD)
        config = Config.parse_double_key(config, 'anchors', 'post_iou_threshold', const.IOU_THRESHOLD)
        config = Config.parse_double_key(config, 'anchors', 'confidence', const.CONFIDENCE)
        config = Config.parse_double_key(config, 'anchors', 'top_k_classes', const.TOP_K_CLASSES)
        
        logging.info("========================")
        logging.info("Config loaded")
        logging.info("========================")
        
        return config
    
    def parse_config_key(config: dict, key: str, default, debug_prefix: str = '') -> dict:
        if (key not in config.keys()):
            logging.warning(f'No {debug_prefix}{key}. Set to default: {default}')
            config[key] = default
        else:
            logging.info(f'{debug_prefix}{key.capitalize()}: {config[key]}')
        return config

    def parse_double_key(config: dict, key1: str, key2: str, default) -> None:
        if (key1 not in config.keys()):
            config[key1] = {}
        Config.parse_config_key(config[key1], key2, default, f'{key1.capitalize()} ')
        return config
