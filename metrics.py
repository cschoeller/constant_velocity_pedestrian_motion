import torch


def avg_disp(y_pred, y_true):
    """ Average displacement error. """
    y_true, masks = y_true

    seq_lengths = masks.sum(1)
    batch_size = len(seq_lengths)

    squared_dist = (y_true - y_pred)**2
    l2_dist = masks * torch.sqrt(squared_dist.sum(2))

    avg_l2_dist = (1./batch_size) * ((1./seq_lengths) * l2_dist.sum(1)).sum()
    return avg_l2_dist.item()


def final_disp(y_pred, y_true):
    """ Final displacement error """
    y_true, masks = y_true

    seq_lengths = masks.sum(1).type(torch.LongTensor) - 1
    batch_size = len(seq_lengths)

    squared_dist = (y_true - y_pred)**2
    l2_dists = masks * torch.sqrt(squared_dist.sum(2))

    disp_sum = l2_dists[:, seq_lengths].sum()
    avg_final_l2_disp = (1./batch_size) * disp_sum

    return avg_final_l2_disp.item()
