from tsai.all import *

my_setup()

dsid = 'NATOPS'
bs = 64
X, y, splits = get_UCR_data(dsid, return_split=False)
print(X.shape)
tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs * 2])

archs = [(FCN, {}), (ResNet, {}), (xresnet1d34, {}), (ResCNN, {}),
         (LSTM, {'n_layers': 1, 'bidirectional': False}), (LSTM, {'n_layers': 2, 'bidirectional': False}),
         (LSTM, {'n_layers': 3, 'bidirectional': False}),
         (LSTM, {'n_layers': 1, 'bidirectional': True}), (LSTM, {'n_layers': 2, 'bidirectional': True}),
         (LSTM, {'n_layers': 3, 'bidirectional': True}),
         (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (XceptionTime, {}), (OmniScaleCNN, {}),
         (mWDN, {'levels': 4})]

results = pd.DataFrame(columns=['arch', 'hyperparams', 'train loss', 'valid loss', 'accuracy', 'time'])
for i, (arch, k) in enumerate(archs):
    model = create_model(arch, dls=dls, **k)
    print(model.__class__.__name__)
    learn = Learner(dls, model, metrics=accuracy)
    learn.create_mbar = False
    start = time.time()
    learn.fit_one_cycle(10, 1e-3)

    elapsed = time.time() - start
    vals = learn.recorder.values[-1]
    results.loc[i] = [arch.__name__, k, vals[0], vals[1], vals[2], int(elapsed)]
    results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)
    display(results)
