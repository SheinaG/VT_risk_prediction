import numpy as np
import torch

import utils.consts as cts
import utils.metrics as metric
from DL.Resnet_pythorch import ResNet18
from DL.dataset import testset

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

task = 'test'
model = ResNet18(img_channels=1, num_classes=2).to(device)
model_path = str(cts.MODELS_DIR / 'model_20220615_180846_2')
model.load_state_dict(torch.load(model_path, map_location=device))
y_true = np.load(cts.TEST_DL_DIR / str("labels_" + task + ".npy"))

testset_d = testset(task=task)
batch_size = 256
testloader = torch.utils.data.DataLoader(testset_d, batch_size=batch_size, shuffle=False, num_workers=10)

model.train(False)
pred_all = []

with torch.no_grad():
    running_vloss = 0.0
    for i, vdata in enumerate(testloader):
        vinputs, vlabels = vdata
        bsn = vinputs.shape[0]
        vinputs = torch.reshape(vinputs, (bsn, 1, -1)).float()
        vinputs = vinputs.to(device)
        vlabels = vlabels.type(torch.LongTensor).to(device)
        voutputs = model(vinputs)
        pred = voutputs.tolist()
        pred_all = pred_all + pred

a = 5

# output = np.load(cts.RESULTS_DIR / 'train_part.npy')
output_exp = np.exp(pred_all)
prob = (output_exp.T / sum(output_exp.T)).T
y_pred = np.argmax(prob.T, axis=0)

metric.model_metrics(prob[:, 1], y_true, y_pred)
