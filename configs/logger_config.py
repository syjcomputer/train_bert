import logging.config as log_config
import logging

def setup_logger(log_file_path, log_file_path2=None):
    log_config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': logging.INFO,
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_file_path,
                'formatter': 'standard',
                'level': logging.INFO,
            },
            'file2': {
                'class': 'logging.FileHandler',
                'filename': log_file_path2,
                'formatter': 'standard',
                'level': logging.INFO,
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': logging.INFO,
                'propagate': False
            },
        },
        'loggers2': {
            '': {  # root logger
                'handlers': ['console', 'file2'],
                'level': logging.INFO,
                'propagate': False
            },
        }

    })