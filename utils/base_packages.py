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
import wfdb
import pandas as pd
import multiprocessing
from numpy import matlib
from collections import defaultdict
import time
from itertools import repeat
import pecg
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
from pecg.ecg import Biomarkers as Obm
from scipy.io import loadmat, savemat
from scipy.stats import norm
import random
from scipy.interpolate import interp1d
from prettytable import PrettyTable
import pickle
import joblib
from datetime import datetime
from pathlib import Path
import shutil
from scipy.fft import fft, ifft, fftshift

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
from skopt.plots import plot_objective, plot_histogram
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')
