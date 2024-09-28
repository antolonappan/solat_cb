# This file contains utility functions that are used in the main script.

import requests
import logging
import numpy as np
from tqdm import tqdm

class Logger:
    def __init__(self, name: str, verbose: bool = False):
        """
        Initializes the logger.
        
        Parameters:
        name (str): Name of the logger, typically the class name or module name.
        verbose (bool): If True, set logging level to DEBUG, otherwise to WARNING.
        """
        self.logger = logging.getLogger(name)
        
        # Configure logging level based on verbosity
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
        
        # Prevent adding multiple handlers to the logger
        if not self.logger.hasHandlers():
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            # Create formatter and add it to the handler
            formatter = logging.Formatter('%(name)s : %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            
            # Add handler to the logger
            self.logger.addHandler(ch)

    def log(self, message: str, level: str = 'info'):
        """
        Logs a message at the specified logging level.
        
        Parameters:
        message (str): The message to log.
        level (str): The logging level (debug, info, warning, error, critical).
        """
        level = level.lower()
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
        else:
            self.logger.info(message)


def inrad(alpha: float) -> float:
    """
    Converts an angle from degrees to radians.

    Parameters:
    alpha (float): The angle in degrees.

    Returns:
    float: The angle in radians.
    """
    return np.deg2rad(alpha)

def cli(cl: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of each element in the input array `cl`.

    Parameters:
    cl (np.ndarray): Input array for which the inverse is calculated.
                     Only positive values will be inverted; zeros and negative values will remain zero.

    Returns:
    np.ndarray: An array where each element is the inverse of the corresponding element in `cl`,
                with zeros or negative values left unchanged.
    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1.0 / cl[np.where(cl > 0)]
    return ret


def download_file(url, filename):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f'Downloading {filename}')
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()