class CBaseException(Exception):
    def __init__(self, msg: str = 'Непредвиденная ошибка.'):
        super().__init__(msg)


class ConfigException(CBaseException):
    def __init__(self, msg: str = ''):
        msg = (
            (
                'Ошибка при работе с файлом конфигурации. Убедитесь, что он '
                'соответствует шаблону. В случае проблем удалите его, чтобы '
                'он создался автоматически.'
            )
            if not msg
            else msg
        )
        super().__init__(msg)


class ImageDifferenceSearcherException(CBaseException):
    def __init__(self, msg: str = ''):
        msg = (
            (
                'Ошибка при поиске отличий. Убедитесь, что '
                'изображения соответствуют требованиям.'
            )
            if not msg
            else msg
        )
        super().__init__(msg)


class FeatureMatcherException(CBaseException):
    def __init__(self, msg: str = ''):
        msg = (
            (
                'Ошибка при сопоставлении особых точек изображений. '
                'Убедитесь, что изображения содержат общую область.'
            )
            if not msg
            else msg
        )
        super().__init__(msg)
