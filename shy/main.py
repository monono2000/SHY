import os
import torch
import warnings
import argparse
import numpy as np
import pickle as pickle
from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if __package__:
    from .dataset import MIMICiiiDataset, MIMICivDataset, transform_and_pad_input
    from .model import SHy
    from .runtime_paths import (
        SAVED_MODELS_DIR,
        TRAINING_LOGS_DIR,
        create_run_dir,
        dataset_dir,
        ensure_runtime_dirs,
    )
    from .training import train
else:
    from dataset import MIMICiiiDataset, MIMICivDataset, transform_and_pad_input
    from model import SHy
    from runtime_paths import (
        SAVED_MODELS_DIR,
        TRAINING_LOGS_DIR,
        create_run_dir,
        dataset_dir,
        ensure_runtime_dirs,
    )
    from training import train

warnings.filterwarnings("ignore")


def save_training_plots(log_path, train_average_loss_per_epoch, test_loss_per_epoch, prediction_loss_per_epoch, r4_list, n4_list):
    try:
        plt.plot(train_average_loss_per_epoch, 'r', label="Train")
        plt.plot(test_loss_per_epoch, 'b', label="Test")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(log_path / "total_loss_plot.svg")
        plt.clf()

        plt.plot(prediction_loss_per_epoch, 'r', label="Test")
        plt.ylabel('Prediction Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(log_path / "prediction_loss_plot.svg")
        plt.clf()

        plt.plot(r4_list, 'r', label="k=20")
        plt.ylabel('Recall')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(log_path / "recall_plot.svg")
        plt.clf()

        plt.plot(n4_list, 'r', label="k=20")
        plt.ylabel('nDCG')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(log_path / "ndcg_plot.svg")
        plt.clf()
    except Exception as exc:
        print(f'Plot generation skipped: {exc}')


if __name__ == '__main__':

    # Settle down all hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=int, default=0, help="GPU index")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")
    parser.add_argument('--dataset_name', type=str, default='MIMIC_III', help="experiment dataset")
    parser.add_argument('--single_dim', type=int, default=32, help="embedding dimension of one ICD-9 level")
    parser.add_argument('--HGNN_dim', type=int, default=256, help="hidden dim in HGNN")
    parser.add_argument('--after_HGNN_dim', type=int, default=128, help="hidden dim after HGNN")
    parser.add_argument('--HGNN_layer_num', type=int, default=2, help="number of HGNN layers")
    parser.add_argument('--nhead', type=int, default=4, help="number of heads in HGNN")
    parser.add_argument('--num_TP', type=int, default=5, help="number of temporal phenotypes")
    parser.add_argument('--n_c', type=int, default=10, help="number of cosine weight vectors")
    parser.add_argument('--hid_state_dim', type=int, default=128, help="temporal phenotype embedding dim")
    parser.add_argument('--key_dim', type=int, default=256,  help="key dim for self attention")
    parser.add_argument('--SA_head', type=int, default=8,  help="number of heads for self-attention")
    parser.add_argument('--dropout', type=float, default=0.001,  help="dropout ratio")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--micro_batch_size', type=int, default=None, help="micro batch size for device-side execution")
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 1, help="number of dataloader workers")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--num_epoch', type=int, default=450, help="number of epochs")
    parser.add_argument('--early_stop_range', type=int, default=10, help="early stop threshold for training process")
    parser.add_argument('--HGNN_model', type=str, default='UniGINConv', help="which hypergraph nn to use")
    parser.add_argument('--temperature', type=float, nargs='+')
    parser.add_argument('--add_ratio', type=float, nargs='+')
    parser.add_argument('--loss_weight', type=float, nargs='+')
    args = parser.parse_args()
    objective_rates = args.loss_weight
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device_idx}" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    if args.micro_batch_size is None:
        args.micro_batch_size = min(args.batch_size, 16) if args.dataset_name == 'MIMIC_IV' else args.batch_size

    # Set random seed and directory information; create directories for saving training results
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    ensure_runtime_dirs()
    current_date_and_time = datetime.now()
    model_directory = f'{current_date_and_time.strftime("%m_%d_%YM%H_%M_%S")}__{args.seed}__{args.dataset_name}'
    model_path = create_run_dir(SAVED_MODELS_DIR, model_directory)
    log_path = create_run_dir(TRAINING_LOGS_DIR, model_directory)

    # Load all data and further preprocess them to get the desirable format
    if args.dataset_name == 'MIMIC_III':
        data_path = dataset_dir(args.dataset_name)
        with open(data_path / 'binary_train_codes_x.pkl', 'rb') as f0:
            binary_train_codes_x = pickle.load(f0)

        with open(data_path / 'binary_test_codes_x.pkl', 'rb') as f1:
            binary_test_codes_x = pickle.load(f1)

        train_codes_y = np.load(data_path / 'train_codes_y.npy')
        train_visit_lens = np.load(data_path / 'train_visit_lens.npy')
        test_codes_y = np.load(data_path / 'test_codes_y.npy')
        test_visit_lens = np.load(data_path / 'test_visit_lens.npy')
        code_levels = np.load(data_path / 'code_levels.npy')
        train_pids = np.load(data_path / 'train_pids.npy')
        test_pids = np.load(data_path / 'test_pids.npy')
        padded_X_train = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
        padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
    else:
        data_path = dataset_dir('MIMIC_IV')
        train_codes_y = np.load(data_path / 'train_codes_y.npy')
        train_visit_lens = np.load(data_path / 'train_visit_lens.npy')
        train_pids = np.load(data_path / 'train_pids.npy')
        test_codes_y = np.load(data_path / 'test_codes_y.npy')
        test_visit_lens = np.load(data_path / 'test_visit_lens.npy')
        test_pids = np.load(data_path / 'test_pids.npy')
        code_levels = np.load(data_path / 'code_levels.npy')

    trans_y_train = torch.as_tensor(train_codes_y, dtype=torch.float32)
    trans_y_test = torch.as_tensor(test_codes_y, dtype=torch.float32)
    class_num = train_codes_y.shape[1]

    # Initialize model and data loaders
    model = SHy(code_levels, args.single_dim, args.HGNN_dim, args.after_HGNN_dim, args.HGNN_layer_num-1, args.nhead, args.num_TP, args.temperature,
                args.add_ratio, args.n_c, args.hid_state_dim, args.dropout, args.key_dim, args.SA_head, args.HGNN_model, device).to(device)
    print(f'Number of parameters of this model: {sum(param.numel() for param in model.parameters())}')
    if args.micro_batch_size < args.batch_size:
        print(f'Using micro-batch size {args.micro_batch_size} inside loader batch size {args.batch_size}.')
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=use_cuda)
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    if args.dataset_name == 'MIMIC_III':
        training_data = MIMICiiiDataset(padded_X_train, trans_y_train, train_pids, train_visit_lens)
        train_loader = DataLoader(training_data, shuffle=True, **loader_kwargs)
        test_data = MIMICiiiDataset(padded_X_test, trans_y_test, test_pids, test_visit_lens)
        test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    else:
        training_data = MIMICivDataset(
            data_path / 'binary_train_x_slices' / 'binary_train_codes_x',
            data_path / 'anchor_train.npy',
            train_visit_lens,
            trans_y_train,
            train_pids,
        )
        train_loader = DataLoader(training_data, shuffle=True, **loader_kwargs)
        test_data = MIMICivDataset(
            data_path / 'binary_test_x_slices' / 'binary_test_codes_x',
            data_path / 'anchor_test.npy',
            test_visit_lens,
            trans_y_test,
            test_pids,
        )
        test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    # Start training
    r2_list, r4_list, n2_list, n4_list, test_loss_per_epoch, train_average_loss_per_epoch, prediction_loss_per_epoch = train(
        model, args.lr, args.num_epoch, train_loader, test_loader, model_path, args.early_stop_range, objective_rates, device, args.micro_batch_size)

    # Save all results
    with open(log_path / 'r2_list.pkl', 'wb') as f101:
        pickle.dump(r2_list, f101)

    with open(log_path / 'r4_list.pkl', 'wb') as f103:
        pickle.dump(r4_list, f103)

    with open(log_path / 'n2_list.pkl', 'wb') as f107:
        pickle.dump(n2_list, f107)

    with open(log_path / 'n4_list.pkl', 'wb') as f109:
        pickle.dump(n4_list, f109)

    with open(log_path / 'train_average_loss_per_epoch.pkl', 'wb') as f112:
        pickle.dump(train_average_loss_per_epoch, f112)

    with open(log_path / 'test_loss_per_epoch.pkl', 'wb') as f113:
        pickle.dump(test_loss_per_epoch, f113)

    with open(log_path / 'prediction_loss_per_epoch.pkl', 'wb') as f114:
        pickle.dump(prediction_loss_per_epoch, f114)

    save_training_plots(log_path, train_average_loss_per_epoch, test_loss_per_epoch, prediction_loss_per_epoch, r4_list, n4_list)
