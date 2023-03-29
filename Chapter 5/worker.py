import numpy as np 
import sys 
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp #A
import os

def square2(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))
    sys.stdout.flush()
    

def square(x):
    return x*x





