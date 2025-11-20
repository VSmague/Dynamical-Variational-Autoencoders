from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # batch: list de tensors [seq_len_i, feat]
    batch = [b for b in batch]  # list de [seq_len, feat]
    padded = pad_sequence(batch, batch_first=False)  # (seq_len_max, batch, feat)
    return padded