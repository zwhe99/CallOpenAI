import logging

class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter with color support for different log levels.
    """
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",   # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",   # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color based on the log level
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)
        record.levelname = f"{color}{levelname}{self.RESET}"
        return super().format(record)

# Configure the logger to use the custom formatter
def setup_colored_logging():
    logger = logging.getLogger()
    handler = logging.StreamHandler()

    formatter = ColoredFormatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add handler to the root logger
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
