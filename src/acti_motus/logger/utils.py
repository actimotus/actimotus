import functools
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

DEFAULT_LOGGING_CONFIG_PATH = resources.files('acti_motus').joinpath('logger/config.json')

# Define a thread-local storage for logging context
log_context = threading.local()


class IdFilter(logging.Filter):
    """
    A logging filter that injects a unique ID into log records.
    This ID is stored in thread-local storage, allowing it to be unique
    for each thread and context.
    """

    def filter(self, record):
        record.id = getattr(log_context, 'id', None)
        return True


@contextmanager
def track_id_context(id=None):
    """Injects the id from our thread-local storage into log records."""
    effective_id = id or str(uuid.uuid4())
    log_context.id = effective_id
    try:
        yield
    finally:
        if hasattr(log_context, 'id'):
            delattr(log_context, 'id')


def traceable_logging(func):
    """
    A decorator that wraps a function to provide traceable logging.
    It injects a unique tracking ID into the logging context for the duration
    of the function call, allowing logs to be associated with specific method calls.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the tracking ID from the method call.
        id = kwargs.pop('id', None)

        # Wrap the method call in our logging context.
        with track_id_context(id=id):
            return func(*args, **kwargs)

    return wrapper


def setup_logging(config: Path):
    if isinstance(config, str):
        config = Path(config)

    if not config.exists():
        raise FileNotFoundError(f"Logging configuration file '{config}' does not exist.")

    with config.open('r') as f:
        config = json.load(f)

    config.setdefault('disable_existing_loggers', False)
    logging.config.dictConfig(config)
