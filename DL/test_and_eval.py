from DL_utiles.base_packages import *
from DL_utiles.helper import *
from utils.consts import *

sys.path.append("/home/sheina/VT_risk_prediction/")

from models.OScnnS import OmniScaleCNN
from models.TCNs import TCN
from models.XceptoinTimeS import XceptionTime
from models.Xception1D import Xception1D
from data.dataset import one_set, overfit_set, t_ansamble_set, all_set

win_len = 180
model = 'Xception1D'
device = 'cuda'
gpu = ''
parser = argparse.ArgumentParser()
parser.add_argument('--downsampling', default=True, type=str2bool, help='Use downsampling to 100 Hz')
run_config = parser.parse_args()
batch_size = 8

if gpu == '':
    device = device
else:
    device = "{}:{}".format(device, gpu)

if model == 'OmniScaleCNN':
    model = OmniScaleCNN(c_in=1, c_out=2, seq_len=run_config.win_len ** 10 * 200)
if model == 'XceptionTime':
    model = XceptionTime(c_in=1, c_out=2)
if model == 'TCN':
    model = TCN(c_in=1, c_out=2, conv_dropout=run_config.conv_dropout, fc_dropout=run_config.fc_dropout)
if model == 'Xception1D':
    model = Xception1D(input_channel=1, n_classes=2)

model_path = str(MODELS_DIR / 'cerulean-salad-194_model_5')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
results = {}

val_set = all_set(task='test', win_len=win_len, run_config=run_config)
test_loader = DataLoader(dataset=val_set, batch_size=batch_size)

pred_all = []

with torch.no_grad():
    running_vloss = 0.0
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        bsn = vinputs.shape[0]
        vinputs = torch.reshape(vinputs, (bsn, 1, -1)).float()
        vinputs = vinputs.to(device)
        vlabels = vlabels.type(torch.LongTensor).to(device)
        voutputs = model(vinputs)
        yv_pred = torch.sigmoid(voutputs)
        pred = yv_pred.tolist()
        pred_all = pred_all + pred

    # wandb.log({"val loss": avg_vloss})
    val_auroc = roc_auc_score(val_set.targets, np.array(pred_all)[:, 1])
    print(val_auroc)
