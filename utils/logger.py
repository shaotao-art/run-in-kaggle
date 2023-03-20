import logging

class Logger:
    def __init__(self, logger_name='logger') -> None:
        self.log_file_path = f'./ckp/{logger_name}.log'

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s--%(message)s')

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        str_handler = logging.StreamHandler()
        str_handler.setFormatter(formatter)
        self.logger.addHandler(str_handler)


    def __str__(self) -> str:
        pass
    
    def log(self, info):
        self.logger.info(info)

    def show_plot(self):
        pass
