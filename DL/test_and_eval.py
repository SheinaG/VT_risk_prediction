from DL_utiles.base_packages import *
from DL_utiles.helper import *
from utils.consts import *

sys.path.append("/home/sheina/VT_risk_prediction/")

from models.OScnnS import OmniScaleCNN
from models.TCNs import TCN
from models.XceptoinTimeS import XceptionTime
from models.Xception1D import Xception1D
from data.dataset import one_set, overfit_set, t_ansamble_set, all_set
from DL_utiles.parse_args import parse_global_args

empty_parser = argparse.ArgumentParser()
parser = parse_global_args(parent=empty_parser)
run_config = parser.parse_args()

run_config = argparse.Namespace(**wandb.config)
set_all_seeds(run_config.seed)

if run_config.gpu == '':
    device = run_config.device
else:
    device = "{}:{}".format(run_config.device, run_config.gpu)

if run_config.model == 'OmniScaleCNN':
    model = OmniScaleCNN(c_in=1, c_out=2, seq_len=run_config.win_len ** 10 * 200)
if run_config.model == 'XceptionTime':
    model = XceptionTime(c_in=1, c_out=2)
if run_config.model == 'TCN':
    model = TCN(c_in=1, c_out=2, conv_dropout=run_config.conv_dropout, fc_dropout=run_config.fc_dropout)
if run_config.model == 'Xception1D':
    model = Xception1D(input_channel=1, n_classes=2)

model = model.to(device)
wandb.watch(model, log='all')
results = {}

with torch.no_grad():
    running_vloss = 0.0
    for i, vdata in enumerate(val_loader):
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
    wandb.log({"val AUROC": val_auroc})
