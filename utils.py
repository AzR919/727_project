"""
Common utility functions

"""

import os
import sys
import datetime

import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager

#--------------------------------------------------------------------------------------------------
# output redirection on system level

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr at the OS level."""
    # Open devnull
    devnull = os.open(os.devnull, os.O_RDWR)

    # Save the actual stdout/stderr file descriptors to restore them later
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)

    try:
        # Flush Python's buffers first
        sys.stdout.flush()
        sys.stderr.flush()

        # Duplicate devnull onto stdout (1) and stderr (2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        yield
    finally:
        # Flush again before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore the original file descriptors
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)

        # Close the temporary file descriptors
        os.close(save_stdout)
        os.close(save_stderr)
        os.close(devnull)

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
