import sys

sys.path.append("/home/sheina/VT_risk_prediction/")

from DL_utiles.base_packages import *
from DL_utiles.helper import *

from models.ROCKET import ROCKET, create_rocket_features
from data.dataset import all_set

set_all_seeds(1)
fs = 200
win_len = 30
device = "cuda"
batch_size = 32
model = ROCKET(c_in=1, seq_len=win_len * 60 * fs, n_kernels=10_000, kss=[7, 9, 11], device=device)

model = model.to(device)
results = {}

train_set = all_set(task='test', win_len=win_len * 60)
val_set = all_set(task='test', win_len=win_len * 60)

timestamp = datetime.now()
train_dl = DataLoader(dataset=train_set, batch_size=batch_size)
test_dl = DataLoader(dataset=val_set, batch_size=batch_size)
train_features = create_rocket_features(train_dl, model)
test_features = create_rocket_features(train_dl, model)
