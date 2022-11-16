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
import joblib
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import pickle
import wfdb
import pandas as pd
import multiprocessing
from numpy import matlib
from collections import defaultdict
import time
from itertools import repeat
import pebm
from pebm import Preprocessing as Pre
from pebm.ebm import FiducialPoints as Fp
from pebm.ebm import Biomarkers as Obm
from scipy.io import loadmat
from scipy.stats import norm
import random
from scipy.interpolate import interp1d

# Machine Learning

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.utils.class_weight as skl_cw
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from skopt.plots import plot_objective
import pymrmr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import joblib
from skopt.plots import plot_objective, plot_histogram
import pickle

# Deep Learning

# warnings.filterwarnings('ignore')
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import os
import pathlib
from pathlib import Path
import wandb
from str2bool import str2bool
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.sampler import DualSampler
