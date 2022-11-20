from utils.base_packages import *

load_dl()
from utils.consts import *

sys.path.append("/home/sheina/VT_risk_prediction/")

from models.OScnnS import OmniScaleCNN
from models.TCNs import TCN
from models.XceptoinTimeS import XceptionTime
from data.dataset import one_set
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
# device = run_config.device
# if run_config.model == 'InceptionTime':
#     model = InceptionTime(c_in=1, c_out=2)
if run_config.model == 'OmniScaleCNN':
    model = OmniScaleCNN(c_in=1, c_out=2, seq_len=run_config.win_len ** 10 * 200)
if run_config.model == 'XceptionTime':
    model = XceptionTime(c_in=1, c_out=2)
if run_config.model == 'TCN':
    model = TCN(c_in=1, c_out=2, conv_dropout=run_config.conv_dropout, fc_dropout=run_config.fc_dropout)
# if run_config.model == 'ResNet':
#     model = ResNet(c_in=1, c_out=2)
# if run_config.model == 'TST':
#     model = TST(c_in=1, c_out=2, seq_len=run_config.win_len**10*200, dropout=run_config.conv_dropout, fc_dropout=run_config.fc_dropout)
# if run_config.model == 'mWDN':
#     model = mWDN(c_in=1, c_out=2, wavelet ='db1', seq_len=run_config.win_len**10*200)


model = model.to(device)
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

train_set = one_set(task='train_part', win_len=run_config.win_len, shuffle=True)
val_set = one_set(task='train_part', win_len=run_config.win_len, shuffle=False)

if run_config.loss == 'wCE':
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, run_config.weight])).to(device)
if run_config.loss == 'AUCMLoss':
    loss_fn = AUCMLoss().to(device)

if run_config.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=run_config.lr)
if run_config.optimizer == 'PESG':
    optimizer = PESG(model, lr=run_config.lr, loss_fn=loss_fn, momentum=0.9, margin=1.0, epoch_decay=0.003,
                     weight_decay=0.0001)

timestamp = datetime.datetime.now()
epoch_gamma = 1


def train_one_epoch(epoch_index):
    train_set.init_epoch(epoch_index)
    sampler = DualSampler(train_set, run_config.batch_size, sampling_rate=0.5)
    train_loader = DataLoader(dataset=train_set, batch_size=run_config.batch_size, sampler=sampler)
    whole_y_pred = np.array([])
    whole_y_t = np.array([])
    running_loss = 0.
    last_loss = 0.
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

    last_loss = running_loss / 1000  # loss per batch
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
    val_set.init_epoch(epoch)
    val_loader = DataLoader(dataset=val_set, batch_size=run_config.batch_size)
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

    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if val_auroc > val_auroc_b:
        val_auroc_b = val_auroc
        model_path = run_config.models_dir + '/' + str(run_config.run_name + '_model_{}'.format(epoch))
        torch.save(model.state_dict(), str(model_path))
