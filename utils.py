"""
Common utility functions

"""

import os
import sys
import datetime

import wandb
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager

#--------------------------------------------------------------------------------------------------
# Helpers

def create_save_str(args):

    now = datetime.datetime.now()
    now = now.strftime("%y-%m-%d_T%H-%M-%S")

    save_str = f"{now}_{args.name_suffix}"

    return save_str

def set_seed(seed):

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)

    # If using CUDA, set seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_reps(dir, wandb_run, rep_tar, rep_out, extra):

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Epoch_{extra}_percs.png")

    plt.plot(rep_tar, color='black', label="target")
    plt.plot(rep_out, color='orange', label="output")
    plt.xlabel("Epoch (*10)")
    plt.ylabel("Percentage")
    plt.ylim(0,1)
    plt.title("Predicted output comparison")
    plt.legend()

    # # Log the second figure and commit the logs for this epoch
    # wandb_run.log({
    #     f"Predicted output comparison": wandb.Image(plt),
    # })

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(dir, losses, extra):

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Epoch_{extra}_loss.png")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss plot")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

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
