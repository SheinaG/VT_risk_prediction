# The main packages used all along the project are loaded here and
# divided by categories
# This script also adds to PATH the different important
# path containing the basic repo directories (utils and parsing)

import sys

sys.path.append("/home/sheina/VT_risk_prediction/")

# General

import os
import os.path
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import pickle
import pandas as pd
import multiprocessing
from numpy import matlib
from collections import defaultdict
import time
from itertools import repeat
from scipy.io import loadmat
from scipy.stats import norm
import random
from scipy.interpolate import interp1d
from prettytable import PrettyTable
import pickle
import joblib
from datetime import datetime
from pathlib import Path



# Deep Learning

# warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import wandb
from str2bool import str2bool
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.sampler import DualSampler
import psutil
import torch
from IPython.display import display
from fastcore.basics import snake2camel
from torch.nn.init import normal_
from torch.nn.utils import weight_norm, spectral_norm
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
