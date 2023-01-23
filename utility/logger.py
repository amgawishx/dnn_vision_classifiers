import logging
import logging.config


class ColoredFormatter(logging.Formatter):
    base = "%(asctime)s: [%(levelname)s] %(message)s"
    red = "\033[91m"
    yellow = "\033[93m"
    green = "\033[92m"
    white = "\033[0m"
    FORMATS = {
        logging.DEBUG: base,
        logging.INFO: green + base + white,
        logging.WARNING: yellow + base + white,
        logging.ERROR: red + base + white,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%I:%M:%S %d-%m-%Y')
        return formatter.format(record)


class LoggedCallable:
    LOGGER = logging

    def __init__(self, f: callable, options: dict):
        settings = {"level": "debug", "with_args": True,
                    "with_kwargs": False, "with_return": True}
        if options:
            settings = {**settings, **options}
        self.logger = getattr(self.LOGGER, settings["level"])
        self.log = "`{name}` has been called"
        if settings["with_args"]:
            self.log += ", with arguments: `{args}`"
        if settings["with_kwargs"]:
            self.log += ", with keyword arguments: `{kwargs}`"
        if settings["with_return"]:
            self.log += ", returned with `{returnable}`"
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            returnable = self.f(*args, **kwargs)
            self.logger(self.log.format(
                name=self.f.__name__,
                args=args, kwargs=kwargs, returnable=returnable))
            return returnable
        except Exception as e:
            self.logger(self.log.format(
                name=self.f.__name__,
                args=args, kwargs=kwargs, returnable=""))
            self.LOGGER.error(
                f"`{type(e).__name__}` has occured while calling `"
                + f"{self.f.__name__}`: {e}")


def logged(options: dict = None) -> callable:
    def wrapper(f: callable) -> callable:
        return LoggedCallable(f, options)
    return wrapper


LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {'default': {'format': ColoredFormatter.base,
                               'datefmt': '%I:%M:%S %d-%m-%Y'},
                   'colored': {'()': ColoredFormatter}},
    'handlers': {
        'console': {'level': 'INFO',
                    'formatter': 'colored',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout'},
        'file': {
            'level': 'DEBUG',
            'formatter': 'default',
            'class': 'logging.FileHandler',
            'filename': 'vcolor.log',
            'mode': 'w'
        }
    },
    'loggers': {
        '': {'handlers': ['console', 'file'],
             'level': 'DEBUG',
             'propagate': False},
    }
}
