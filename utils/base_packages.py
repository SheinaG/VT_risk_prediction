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
import patlib
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

# Machine Learning

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.utils.class_weight as skl_cw
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from skopt.plots import plot_objective
import pymrmr

# Deep Learning

# warnings.filterwarnings('ignore')
