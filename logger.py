import logging
import os


class Logger:
    """
    Класс для упрощенной работы с логгированием.
    """

    def __init__(self, name, filename='app.log', level=logging.DEBUG):
        """
        Инициализирует объект логгера.

        :param name: Имя логгера (обычно __name__).
        :param filename: Имя файла для записи логов.
        :param level: Уровень логгирования (DEBUG, INFO, WARNING, ERROR,
        CRITICAL).
        """

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(
            os.path.join(log_dir, filename), mode='a'
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                      '- %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        """Логирует сообщение уровня DEBUG."""
        self.logger.debug(message)

    def info(self, message):
        """Логирует сообщение уровня INFO."""
        self.logger.info(message)

    def warning(self, message):
        """Логирует сообщение уровня WARNING."""
        self.logger.warning(message)

    def error(self, message):
        """Логирует сообщение уровня ERROR."""
        self.logger.error(message)

    def critical(self, message):
        """Логирует сообщение уровня CRITICAL."""
        self.logger.critical(message)
