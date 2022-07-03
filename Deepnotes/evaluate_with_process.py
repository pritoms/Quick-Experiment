# `evaluate_with_processes.py`

from process_handler import Process, ProcessManager
from bytes_tokenizer import encode, decode, tokenizer
from text_dataset import TextDataset
from transformer_model import create_model, create_mask
from checkpoint_manager import *
from train_eval_utils import *
from modeling_utils import *
from Learning Experiment import LearningExperiment

import numpy as np
import time
import torch
import os
import shutil
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Any
from torch.nn import Module

def main():
    cmd = ['python', 'evaluate.py']
    # Create an instance of the Process class and create a subprocess.
    p = Process(cmd)
    # Start the subprocess and wait for it to finish.

    with ProcessManager([p]):
        print('Process pid: %d' % p.pid)
        while p.is_alive:
            time.sleep(.1)
            print('.', end='')
            sys.stdout.flush()
        print('\nProcess finished with return code %d' % p.returncode)

if __name__ == '__main__':
    main()