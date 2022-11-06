import utils.consts as cts
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

import utils.consts as cts

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

model = ResNet34(img_channels=1, num_classes=3).to(device)
model_path = str(cts.MODELS_DIR / 'model_20220704_131938_10')
model.load_state_dict(torch.load(model_path, map_location=device))

testset_d = multi_set(task='test')
y_true = testset_d.labels
batch_size = 256
testloader = torch.utils.data.DataLoader(testset_d, batch_size=batch_size, shuffle=False, num_workers=10)

model.train(False)
pred_all = []

with torch.no_grad():
    running_vloss = 0.0
    for i, tdata in enumerate(testloader):
        tinputs, tlabels = tdata
        bsn = tinputs.shape[0]
        tinputs = torch.reshape(tinputs, (bsn, 1, -1)).float()
        tinputs = tinputs.to(device)
        tlabels = tlabels.type(torch.LongTensor).to(device)
        toutputs = model(tinputs)
        pred = toutputs.tolist()
        pred_all = pred_all + pred

# output = np.load(cts.RESULTS_DIR / 'train_part.npy')
output_exp = np.exp(pred_all)
prob = (output_exp.T / sum(output_exp.T)).T
y_pred = np.argmax(prob.T, axis=0)

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
c_m = confusion_matrix(y_true, y_pred)
print(cohen_kappa_score(y_true, y_pred))
print(c_m)
