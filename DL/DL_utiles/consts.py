TEST_DL_DIR = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/test')
TRAIN_DL_DIR = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train')


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
