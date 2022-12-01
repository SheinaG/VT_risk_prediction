from DL.DL_utiles.base_packages import *



def parse_global_args(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    # general
    repo_root = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)))
    dataset_root = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/')
    output_folder = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/models/')
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    parser.add_argument('--gpu', default='0', type=str, help='specific gpu number to use {0,1,2...}')
    parser.add_argument('--repo_root', default=repo_root, type=Path, help='repository main dir')
    parser.add_argument('--models_dir', default=output_folder, type=Path, help='output dir')
    parser.add_argument('--dataset_dir', default=dataset_root, type=Path, help='dataset path')
    parser.add_argument('--run-name', default='', type=str, help='placeholder for wandb run name')
    parser.add_argument('--seed', default=0, type=int, help='random seed for everything')
    parser.add_argument('--device', default=dev, type=str, help='which device to train')
    parser.add_argument('--train', default='overfit', type=str, help='to train or to evaluate')
    parser.add_argument('--model', default='XceptionTime', type=str,
                        choices=['XceptionTime', 'InceptionTime', 'TCN', 'ResNet', 'OmniScaleCNN', 'LSTM_FCN', 'LSTM',
                                 'RNN'], help='which model to train')
    parser.add_argument('--conv_dropout', default=0, type=float, help='how mach dropout to use')
    parser.add_argument('--fc_dropout', default=0, type=float, help='how mach dropout to use')
    parser.add_argument('--use_sampler', default=True, type=str2bool, help='if to use sampler or not')

    # data

    parser.add_argument('--win_len', default=6, type=int, choices=[1, 6, 30, 60, 180, 360])
    parser.add_argument('--data-aug', default=False, type=str2bool, help='Use data augmentations')

    # torch
    parser.add_argument('--workers', default=0, type=int, help='number of workers in torch')

    # learning
    parser.add_argument('--loss', default='AUCMLoss', type=str, choices=['CE', 'wCE', 'focal', 'AUCMLoss'],
                        help='loss to use')
    parser.add_argument('--weight', default=36, type=int, help='The weight of the positive class')
    parser.add_argument('--epochs', type=int, default=10000, help='Total number of epochs')
    # parser.add_argument('--early-stop-patience', type=int, default=10, help='')
    # parser.add_argument('--val-step-every', type=int, default=1, help='run validation set every x number of epochs')
    parser.add_argument('--optimizer', default='PESG', type=str, choices=['AdamW', 'sgd', 'Adam', 'PESG'],
                        help='which optimizer to use')
    # parser.add_argument('--scheduler_patience', default=5, type=int, help='patience of reduce lr on plateau')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    # parser.add_argument('--weight_decay', type=float, default=1e-4, help='Regularization term')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--size', type=int, default=16, help='overfit size')
    parser.add_argument('--batch-size-val', type=int, default=4, help='validation batch size')
    parser.add_argument('--batch-size-test', type=int, default=4, help='test batch size')

    # TCN model:
    parser.add_argument('--activation', type=str, default='leakyRelu', help='which activation function to use')
    parser.add_argument('--n_layers', type=int, default=4, help='how many layers are in the model')
    parser.add_argument('--ni', type=int, default=25, help='how many channels are in the model')
    parser.add_argument('--ks', type=int, default=17, help='filter size')

    return parser
