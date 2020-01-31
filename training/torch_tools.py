from itertools import zip_longest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from util.misc import assert_shape


def dataset2numpy(dataset, fields):
    """
    Converts a PyTorch Dataset to a numpy array.

    Parameters:
        fields: list of fields to return from the full dataset.
    """

    loader = DataLoader(dataset, batch_size=len(dataset) // 8, num_workers=8)
    parts = []
    for l in loader:
        parts.append(l)

    return [np.concatenate([p[f].numpy() for p in parts], axis=0) for f in fields]


def torch_predict(model, input, batch_size=None, device='cuda'):
    """

    :param model: PyTorch Model(nn.Module)
    :param input: a numpy array or a PyTorch dataloader
    :param batch_size: if input was a numpy array, this is the batch size used for evaluation
    :return:
    """
    model.eval()

    if isinstance(input, np.ndarray):
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input).to(device)), batch_size)
        needs_move = False
    elif isinstance(input, torch.Tensor):
        data_loader = DataLoader(TensorDataset(input.to(device)), batch_size)
        needs_move = False
    else:
        data_loader = input
        needs_move = True

    result = []
    with torch.no_grad():
        for batch in data_loader:
            if needs_move:
                if isinstance(batch, (list, tuple, map)):
                    batch = map(lambda x: x.to(device), batch)
                elif isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                else:
                    batch = batch.to(device)

            if isinstance(batch, (list, tuple, map)):
                pred = model(*batch)
            elif isinstance(batch, dict):
                pred = model(**batch)
            else:
                pred = model(batch)

            if isinstance(pred, (list, tuple, map)):
                result.append([x.cpu().numpy() for x in pred])
            else:
                result.append(pred.cpu().numpy())

            del pred

    if isinstance(result[0], list):
        out = []
        for i in range(len(result[0])):
            out.append(np.concatenate([x[i] for x in result]))
        result = out
    else:
        result = np.concatenate(result)

    return result


def eval_results(pred3d, gt3d, joint_set, verbose=True, pck_threshold=150, pctiles=[99]):
    """
    Evaluates the results by printing various statistics. Also returns those results.
    Poses can be represented either in hipless 16 joints or 17 joints with hip format.
    Order is MuPo-TS order in all cases.

    Parameters:
        pred3d: dictionary of predictions in mm, seqname -> (nSample, [16|17], 3)
        gt3d: dictionary of ground truth in mm, seqname -> (nSample, [16|17], 3)
        joint_set; JointSet instance describing the order of joints
        verbose: if True, a table of the results is printed
        pctiles: list of percentiles of the errors to calculate
    Returns:
        sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles
    """

    has_hip = list(pred3d.values())[0].shape[1] == joint_set.NUM_JOINTS  # whether it contains the hip or not

    if has_hip:
        common14_joints = joint_set.TO_COMMON14
    else:
        common14_joints = np.array(joint_set.TO_COMMON14[1:]).copy()  # a copy of original list
        hip_ind = joint_set.index_of('hip')
        common14_joints[common14_joints > hip_ind] -= 1

    sequence_mpjpes = {}
    sequence_pcks = {}
    sequence_aucs = {}
    sequence_common14_pcks = {}  # pck for the common 14 joints (used by Mehta et al.)
    sequence_pctiles = {}
    all_errs = []

    for k in sorted(pred3d.keys()):
        pred = pred3d[k]
        gt = gt3d[k]

        assert pred.shape == gt.shape, "Pred shape:%s, gt shape:%s" % (pred.shape, gt.shape)
        assert (not has_hip and pred.shape[1:] == (joint_set.NUM_JOINTS - 1, 3)) or \
               (has_hip and pred.shape[1:] == (joint_set.NUM_JOINTS, 3)), \
            "Unexpected shape:" + str(pred.shape)

        errs = np.linalg.norm(pred - gt, axis=2, ord=2)  # (nSample, nJoints)

        sequence_pctiles[k] = np.nanpercentile(errs, pctiles)
        sequence_pcks[k] = np.nanmean((errs < pck_threshold).astype(np.float64))
        pck_curve = []
        for t in range(0, 151, 5):  # go from 0 to 150, 150 inclusive
            pck_curve.append(np.mean((errs < t).astype(np.float64)))
        sequence_aucs[k] = np.mean(pck_curve)

        sequence_common14_pcks[k] = np.nanmean((errs[:, common14_joints] < pck_threshold).astype(np.float64))
        sequence_mpjpes[k] = np.nanmean(errs)

        # Adjusting results for missing hip
        if not has_hip:
            N = float(joint_set.NUM_JOINTS)
            sequence_pcks[k] = sequence_pcks[k] * ((N - 1) / N) + 1. / N
            sequence_aucs[k] = sequence_aucs[k] * ((N - 1) / N) + 1. / N
            sequence_common14_pcks[k] = sequence_common14_pcks[k] * ((N - 1) / N) + 1. / N
            sequence_mpjpes[k] = sequence_mpjpes[k] * ((N - 1) / N)

        all_errs.append(errs)

    all_errs = np.concatenate(all_errs)  # errors per joint, (nPoses, nJoints)
    joint_mpjpes = np.nanmean(all_errs, axis=0)
    joint_pctiles = np.nanpercentile(all_errs, pctiles, axis=0)

    num_joints = joint_set.NUM_JOINTS if has_hip else joint_set.NUM_JOINTS - 1
    assert_shape(all_errs, (None, num_joints))
    assert_shape(joint_mpjpes, (num_joints,))
    assert_shape(joint_pctiles, (len(pctiles), num_joints))

    if verbose:
        joint_names = joint_set.NAMES.copy()
        if not has_hip:
            joint_names = np.delete(joint_names, joint_set.index_of('hip'))  # remove root

        # Index of the percentile that will be printed. If 99 is calculated it is selected,
        # otherwise the last one
        pctile_ind = len(pctiles) - 1
        if 99 in pctiles:
            pctile_ind = pctiles.index(99)

        print(" ----- Per sequence and joint errors in millimeter on the validation set ----- ")
        print(" %s       %6s      %5s   %6s   \t %22s  %6s     %6s" % ('Sequence', 'Avg', 'PCK', str(pctiles[pctile_ind]) + '%', '',
                                                                       'Avg', str(pctiles[pctile_ind]) + '%'))
        for seq, joint_id in zip_longest(sorted(pred3d.keys()), range(num_joints)):
            if seq is not None:
                seq_str = " %-8s:   %6.2f mm   %4.1f%%   %6.2f mm\t " \
                          % (str(seq), sequence_mpjpes[seq], sequence_pcks[seq] * 100, sequence_pctiles[seq][pctile_ind])
            else:
                seq_str = " " * 49

            if joint_id is not None:
                print('%s%15s (#%2d):  %6.2f mm   %6.2f mm ' % (seq_str, joint_names[joint_id], joint_id,
                                                                joint_mpjpes[joint_id], joint_pctiles[pctile_ind, joint_id]))
            else:
                print(seq_str)

        mean_sequence_err = np.mean(list(sequence_mpjpes.values()))
        print("\nMean absolute error: %6.2f mm" % mean_sequence_err)

    return sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_mpjpes, joint_pctiles
