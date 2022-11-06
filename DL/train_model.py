from datetime import datetime

import consts as cts
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from DL.Resnet_pythorch import ResNet18
from DL.dataset import trainset, testset

batch_size = 64

if torch.cuda.is_available():
    dev = "cuda:2"
else:
    dev = "cpu"
device = torch.device(dev)

trainset_d = trainset()
testset_d = testset()
print('loading')
trainloader = torch.utils.data.DataLoader(trainset_d, batch_size=batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset_d, batch_size=batch_size, shuffle=False, num_workers=2)

model = ResNet18(img_channels=1, num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainloader):
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
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        torch.cuda.empty_cache()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(trainloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 20

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)
    pred_all = []
    print('evaluating')
    with torch.no_grad():
        running_vloss = 0.0
        for i, vdata in enumerate(testloader):
            vinputs, vlabels = vdata
            bsn = vinputs.shape[0]
            vinputs = torch.reshape(vinputs, (bsn, 1, -1)).float()
            vinputs = vinputs.to(device)
            vlabels = vlabels.type(torch.LongTensor).to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            pred = voutputs.tolist()
            pred_all = pred_all + pred

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        np.save(cts.RESULTS_DIR / 'train_part.npy', pred_all)
        model_path = cts.MODELS_DIR / 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), str(model_path))

    epoch_number += 1
