import logging
import os
import functools

console_log_handler = logging.StreamHandler()  # Console handler
file_log_handler = logging.FileHandler('particleFilterTeam.log')  # File handler
console_log_handler.setLevel(logging.DEBUG)
file_log_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_log_handler.setFormatter(c_format)
file_log_handler.setFormatter(f_format)


class DeferredFileHandler(logging.Handler):
    def __init__(self, name, formatter=logging.Formatter('%(message)s')):
        super().__init__()
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)
        self.filepath = os.path.join(_logDir, name)
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))

    def flush(self):
        with open(self.filepath, 'a') as f:
            f.write('\n'.join(self.buffer) + '\n')
        self.buffer = []


def __createEmptyLogDir(baseDir):
    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    if not os.listdir(baseDir):  # Check if directory is empty
        return baseDir

    # Find the next available directory name
    counter = 1
    while True:
        new_dir = os.path.join(baseDir, f'{counter:02d}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            break
        counter += 1

    return new_dir

_logDir = __createEmptyLogDir('particle_filter')
