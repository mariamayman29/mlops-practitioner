import logging
import sys
from datetime import datetime
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="system")

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        log_record['timestamp'] = datetime.utcnow().isoformat() + "Z"
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['correlation_id'] = correlation_id_var.get()


def setup_logging():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)

    log_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter()
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)