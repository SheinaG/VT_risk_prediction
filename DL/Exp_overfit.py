from DL_utiles.base_packages import *
from DL_utiles.helper import *
from utils.consts import *

sys.path.append("/home/sheina/VT_risk_prediction/")

from models.OScnnS import OmniScaleCNN
from models.TCNs import TCN
from models.XceptoinTimeS import XceptionTime
from data.dataset import one_set, overfit_set
from DL_utiles.parse_args import parse_global_args


empty_parser = argparse.ArgumentParser()
parser = parse_global_args(parent=empty_parser)
run_config = parser.parse_args()

wandb.init(project="VT_det", entity="sheina", config=run_config)
run_config = argparse.Namespace(**wandb.config)
run_config.run_name = wandb.run.name
set_all_seeds(run_config.seed)

fs = 200

if run_config.gpu == '':
    device = run_config.device
else:
    device = "{}:{}".format(run_config.device, run_config.gpu)

if run_config.model == 'OmniScaleCNN':
    model = OmniScaleCNN(c_in=1, c_out=2, seq_len=run_config.win_len ** 10 * 200)
if run_config.model == 'XceptionTime':
    model = XceptionTime(c_in=1, c_out=2, ks=run_config.ks, nf=run_config.ni, n_layers=run_config.n_layers)
if run_config.model == 'TCN':
    model = TCN(c_in=1, c_out=2, layers=run_config.n_layers * [run_config.ni], ks=run_config.ks,
                conv_dropout=run_config.conv_dropout, fc_dropout=run_config.fc_dropout,
                activation=run_config.activation)

model = model.to(device)
wandb.watch(model, log='all')
results = {}

if run_config.batch_size == 0:
    if run_config.win_len == 1:
        run_config.batch_size = 32
    if run_config.win_len == 6:
        run_config.batch_size = 32
    if run_config.win_len == 30:
        run_config.batch_size = 8
    if run_config.win_len == 60:
        run_config.batch_size = 4
    if run_config.win_len == 180:
        run_config.batch_size = 1

train_set = overfit_set(task='train_part', win_len=run_config.win_len, size=run_config.size)
train_set.init_epoch(epoch_idx=0)
train_loader = DataLoader(dataset=train_set, batch_size=run_config.size * 2)
# val_set = one_set(task='val', win_len=run_config.win_len, shuffle=False)

if run_config.loss == 'wCE':
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, run_config.weight])).to(device)
if run_config.loss == 'AUCMLoss':
    loss_fn = AUCMLoss().to(device)

if run_config.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=run_config.lr)
if run_config.optimizer == 'PESG':
    optimizer = PESG(model, lr=run_config.lr, loss_fn=loss_fn, momentum=0.9, margin=1.0, epoch_decay=0.003,
                     weight_decay=0.0001)

timestamp = datetime.now()
epoch_gamma = 1


def train_one_epoch(epoch_index):

    running_loss = 0.
    pred_all = []
    lab_all = []
    pred_epoch = []
    lab_epoch = []

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        bsn = inputs.shape[0]
        inputs = torch.reshape(inputs, (bsn, 1, -1)).float()
        inputs = inputs.to(device)

        labels = labels.type(torch.LongTensor).to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        y_pred = torch.sigmoid(outputs)
        pred = y_pred.tolist()
        pred_all = pred_all + pred
        lab_all = lab_all + labels.tolist()
        pred_epoch = pred_epoch + pred
        lab_epoch = lab_epoch + labels.tolist()

        # Compute the loss and its gradients
        loss = loss_fn(y_pred, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / (i + 1)  # loss per batch
    print('  batch {} loss: {}'.format(i + 1, last_loss))
    wandb.log({"train batch loss": last_loss})
    wandb.log({"train batch AUC": roc_auc_score(np.array(lab_all), np.array(pred_all)[:, 1])})
    print(roc_auc_score(np.array(lab_all), np.array(pred_all)[:, 1]))
    running_loss = 0.
    pred_all = []
    lab_all = []

    return last_loss, pred_epoch, lab_epoch


val_auroc_b = 0
EPOCHS = run_config.epochs
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, tp_e, lp_e = train_one_epoch(epoch)

    # We don't need gradients on to do reporting
    model.train(False)
    pred_all = []
    print('evaluating')

    with torch.no_grad():
        running_vloss = 0.0
        for i, vdata in enumerate(train_loader):
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
    val_auroc = roc_auc_score(train_set.targets, np.array(pred_all)[:, 1])
    wandb.log({"val AUROC": val_auroc})
    if val_auroc == 1:
        run_config.size = run_config.size * 2
        train_set = overfit_set(task='train_part', win_len=run_config.win_len, size=run_config.size)
        train_set.init_epoch(epoch_idx=0)
        train_loader = DataLoader(dataset=train_set, batch_size=run_config.size * 2)
        print(run_config.size)

    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if val_auroc > val_auroc_b:
        val_auroc_b = val_auroc
        wandb.log({"best val AUROC": val_auroc_b})
    #     model_path = run_config.models_dir + '/' + str(run_config.run_name + '_model_{}'.format(epoch))
    #     torch.save(model.state_dict(), str(model_path))
