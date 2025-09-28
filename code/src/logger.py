import logging

from src.config import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs if root logger is configured

    if logger.handlers:
        return logger

    file_path = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(str(file_path))

    file_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    return logger
