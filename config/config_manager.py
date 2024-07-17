import os
import traceback

import yaml

from exceptions import ConfigException
from logger import Logger

logger = Logger(__name__)

DIFFERENCE_SEARCHER_SIGNAL_NAME = 'difference_searcher_progress'
STITCHER_SIGNAL_NAME = 'stitcher_progress'

DEFAULT_CONFIG = {
    'feature_matcher': {
        'detector': 'superpoint',
        'ransac_reproj_threshold': 10.0,
        'match_conf': 0.5,
        'max_features': 300,
        'flann_index_params': {
            'algorithm': 1,
            'trees': 5
        },
        'flann_search_params': {
            'checks': 50
        },
        'super_point': {
            'weights_path': 'backend/models/superpoint_v1.pth',
            'nms_dist': 8,
            'conf_thresh': 0.002,
            'nn_thresh': 0.9,
        },
        'img_size_scale': 0.5
    },
}


def check_range(key, value, min_val, max_val, error_messages):
    if not isinstance(value, int):
        error_messages.append(f"{key} должен быть целым числом")
    elif value < min_val or value > max_val:
        error_messages.append(
            f"{key} должен быть в диапазоне [{min_val}, {max_val}]")


def _verify_config(cfg):
    """
    Проверка конфигурации на наличие необходимых ключей.
    """
    error_messages = []

    def _check_keys(data, keys, prefix=''):
        for key, value in keys.items():
            if isinstance(value, dict):
                _check_keys(data.get(key, {}), value, f'{prefix}{key}.')
            else:
                if key not in data:
                    error_messages.append(f'Отсутствует ключ: {prefix}{key}')

    _check_keys(cfg, DEFAULT_CONFIG)

    if error_messages:
        logger.error('В конфиге отсутствуют обязательные параметры:\n'
                     + '\n'.join(error_messages))
        raise ConfigException('\n'.join(error_messages))


def read_config(config_path: str = 'config/config.yml') -> dict:
    """
    Чтение конфигурации из YAML файла.

    Args:
        config_path: Путь к YAML файлу.

    Returns:
        dict: Словарь с настройками.
    """
    if not os.path.exists(config_path):
        create_config(config_path)
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError:
            logger.error(traceback.format_exc())
            raise ConfigException('Ошибка при чтение конфига.')

    _verify_config(config)
    return config


def save_config(config: dict, config_path: str = 'config.yml'):
    """
    Сохранение конфигурации в YAML файл.

    Args:
        config: Словарь с настройками.
        config_path: Путь к YAML файлу.
    """
    with open(config_path, 'w') as f:
        try:
            yaml.dump(config, f)
        except yaml.YAMLError:
            logger.error(traceback.format_exc())
            raise ConfigException('Ошибка при сохранении конфига.')


def create_config(config_path):
    """Создает папку 'config' и файл 'config.yml' с дефолтными параметрами."""

    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, indent=2, default_flow_style=False)
