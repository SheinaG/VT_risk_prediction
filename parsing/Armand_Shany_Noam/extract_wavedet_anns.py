from base_parser import *

start = 1001
end = 2893
res = sio.loadmat(
    cts.DATA_DIR / "Annotations" / "UVAFDB" / "wavedet" / ("all_anns_" + str(start) + "_" + str(end) + ".mat"))

for i in range(len(res['final_cell'])):
    if i + start < 1096:
        continue
    ann = res['final_cell'][i][0][0].astype(int)
    if len(ann) > 0:
        ann = ann[ann > 0]
        ann = np.delete(ann, np.where(np.diff(ann) < 0)[0])
        id = str(i + start).zfill(4)
        print(id)
        final_path = cts.DATA_DIR / "Annotations" / "UVAFDB" / "wavedet" / (id + '.wavedet')
        wfdb.wrann(id, 'wavedet', ann, symbol=['q'] * len(ann))
        shutil.move(id + '.wavedet', final_path)
