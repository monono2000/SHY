import torch
import numpy as np
import torch.utils.data as data
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


# Prepare padded hypergraphs for batched input
def transform_and_pad_input(x):
    tempX = []
    for ele in x:
        tempX.append(torch.as_tensor(ele, dtype=torch.float32))
    x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
    return x_padded


class MIMICiiiDataset(data.Dataset):
    def __init__(self, patient, label, pids, visit_len):
        self.patient = patient
        self.label = label
        self.pids = pids
        self.visit_len = visit_len

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.patient[idx], self.label[idx], self.pids[idx], self.visit_len[idx]


class MIMICivDataset(data.Dataset):
    def __init__(self, x_prefix, anchor_path, v_lens, labels, pids):
        self.x_prefix = Path(x_prefix)
        self.anchor_path = Path(anchor_path)
        self.v_lens, self.labels, self.pids = v_lens, labels, pids
        anchor_shape = np.load(self.anchor_path, mmap_mode='r').shape
        self.max_visit_len, self.code_num = anchor_shape

    def __len__(self):
        return len(self.pids)

    def transform_and_pad_input(self, source):
        source_tensor = torch.from_numpy(source).to(torch.float32)
        padded = source_tensor.new_zeros((self.max_visit_len, self.code_num))
        padded[:source_tensor.shape[0]] = source_tensor
        return torch.transpose(padded, 0, 1)

    def __getitem__(self, idx):
        source = np.load(self.x_prefix.parent / f'{self.x_prefix.name}_{idx}.npy')
        padded = self.transform_and_pad_input(source)
        return padded, self.labels[idx], self.pids[idx], self.v_lens[idx]

