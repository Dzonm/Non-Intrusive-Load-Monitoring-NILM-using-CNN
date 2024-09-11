import logging
import time

class Logger():
    """Logger class for NILM project."""
    
    def __init__(
            self, lowest_level: str = 'info', log_file_name: str = None, append: bool = False
        ) -> None:
        """Initializes the logger."""
        if log_file_name is None:
            log_file_name = '{}.log'.format(time.strftime("%Y-%m-%d-%H:%M:%S").replace(':','-'))

        mode = 'a' if append else 'w'
        with open(log_file_name, mode=mode, encoding='utf-8'):
            pass

        self.root_logger = logging.getLogger()

        log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')

        self.file_handler = logging.FileHandler(log_file_name)
        self.file_handler.setFormatter(log_formatter)
        self.root_logger.addHandler(self.file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.root_logger.addHandler(console_handler)

        # Set lowest-severity log message logger will handle.
        if lowest_level == 'debug':
            self.root_logger.setLevel(logging.DEBUG)
        elif lowest_level == 'warning':
            self.root_logger.setLevel(logging.WARNING)
        elif lowest_level == 'critical':
            self.root_logger.setLevel(logging.CRITICAL)
        else:
            self.root_logger.setLevel(logging.INFO)

        # Disable debug messages from the following modules.
        disable_debug_modules = [
            'matplotlib',
            'matplotlib.font',
            'matplotlib.pyplot',
            'matplotlib.font_manager',
            'PIL'
        ]
        for module in disable_debug_modules:
            logger = logging.getLogger(module)
            logger.setLevel(logging.INFO)

    def log(self, message: str, level: str = 'info') -> None:
        """Logs a message with the specified severity level."""
        if level == 'debug':
            self.root_logger.debug(message)
        elif level == 'warning':
            self.root_logger.warning(message)
        elif level == 'critical':
            self.root_logger.critical(message)
        else:
            self.root_logger.info(message)
            
    def closefile(self) -> None:
        """Closes the log file and cleans up the logger."""
        # Close and remove the file handler
        if self.file_handler:
            self.file_handler.close()
            self.root_logger.removeHandler(self.file_handler)

        # Shut down the logging system
        logging.shutdown()
