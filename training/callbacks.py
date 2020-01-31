import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from training.torch_tools import dataset2numpy
from training.torch_tools import eval_results, torch_predict
from util.pose import remove_root


class BaseCallback(object):

    def on_itergroup_end(self, iter_cnt, epoch_loss):
        pass

    def on_epoch_end(self, model, epoch, epoch_loss, optimizer, epoch_vals):
        pass


class BaseMPJPECalculator(BaseCallback):
    """
    Base class for calculating and displaying MPJPE stats, grouped by something (sequence most of the time).
    """
    PCK_THRESHOLD = 150

    def __init__(self, data_3d_mm, joint_set, post_process3d=None, prefix='val'):
        """

        :param data_3d_mm: dict, group_name-> ndarray(n.Poses, nJoints, 3). The ground truth poses in mm.
        """
        self.prefix = prefix
        self.pctiles = [5, 10, 50, 90, 95, 99]

        self.data_3d_mm = data_3d_mm

        self.is_absolute = _sample_value(self.data_3d_mm).shape[1] == joint_set.NUM_JOINTS
        self.num_joints = joint_set.NUM_JOINTS if self.is_absolute else joint_set.NUM_JOINTS - 1

        self.joint_set = joint_set
        self.post_process3d = post_process3d
        self.sequences = sorted(list(data_3d_mm.keys()))

    def on_epoch_end(self, model, epoch, epoch_loss, optimizer, epoch_vals):
        self.eval(model)


    def eval(self, model=None):
        losses, preds = self.pred_and_calc_loss(model)
        losses = np.concatenate([losses[seq] for seq in self.sequences])
        self.val_loss = np.nanmean(losses)
        self.losses_to_log = {self.prefix + '_loss': self.val_loss}

        self.losses = losses
        self.preds = preds

        # Assuming hip is the last component
        if self.is_absolute:
            self.losses_to_log[self.prefix + '_abs_loss'] = np.nanmean(losses[:, -3:])
            self.losses_to_log[self.prefix + '_rel_loss'] = np.nanmean(losses[:, :-3])
        else:
            self.losses_to_log[self.prefix + '_rel_loss'] = self.val_loss

        assert self.pctiles[-1] == 99, "Currently the last percentile is hardcoded to be 99 for printing"

        sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles = \
            eval_results(preds, self.data_3d_mm, self.joint_set, pctiles=self.pctiles, verbose=True)

        # Calculate relative error
        if self.is_absolute:
            rel_pred = {}
            rel_gt = {}
            for seq in preds:
                rel_pred[seq] = remove_root(preds[seq], self.joint_set.index_of('hip'))
                rel_gt[seq] = remove_root(self.data_3d_mm[seq], self.joint_set.index_of('hip'))
            rel_mean_error, _, _, _, _ = eval_results(rel_pred, rel_gt, self.joint_set, verbose=False)
            rel_mean_error = np.mean(list(rel_mean_error.values()))
            print("Root relative error: %.2f mm" % rel_mean_error)
            self.rel_mean_error = rel_mean_error
            self.losses_to_log[self.prefix + '_rel_error'] = rel_mean_error

        self.mean_sequence_mpjpe = np.mean(list(sequence_mpjpes.values()))
        self.mean_sequence_pck = np.mean(list(sequence_pcks.values()))
        self.losses_to_log[self.prefix + '_err'] = self.mean_sequence_mpjpe
        self.losses_to_log[self.prefix + '_pck'] = self.mean_sequence_pck

        return sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles

    def pred_and_calc_loss(self, model):
        """
        Subclasses must implement this method. It calculates the loss
        and the predictions of the current model.

        :param model: model received in the on_epoch_end callback
        :return: (loss, pred) pair, each is a dictionary from sequence name to loss or prediction
        """
        raise NotImplementedError()


def _sample_value(dictionary):
    """ Selects a value from a dictionary, it is always the same element. """
    return list(dictionary.values())[0]


class LogAllMillimeterError(BaseMPJPECalculator):
    """
    This callback evaluates the model on every epoch, both actionwise and jointwise.
    Results can be saved optionally.

    PCK, mpjpe calculated over all joints (including hip, which has zero error)
    """

    @staticmethod
    def eval_model(model, dataset, loss, post_process3d):
        """
        Evaluates a model on a dataset and prints statistics on the screen. Parameters are
        the same as the ones in `from_dataset`.
        """
        l = LogAllMillimeterError.from_dataset(dataset, loss, post_process3d, model=model)
        l.eval()
        return l

    @staticmethod
    def from_dataset(dataset, loss, post_process3d=None, model=None, prefix='val'):
        """

        :param dataset: torch Dataset that stores coordinates in poses2d, poses3d and has an index for identification
        :param joint_set: JointSet instance d
        :param post_process3d: 3D unnormalisation function, input is (nPoses, nJoints*3), output is (nPoses, nJoints, 3)
        :param csv: None or path to a csv file, results are also written to the given file if not none
        :param model: model to evaluate, if none the model in training is used.
        """
        sequences = np.unique(dataset.index.seq).tolist()
        preprocessed2d, preprocessed3d = dataset2numpy(dataset, ['pose2d', 'pose3d'])

        # need to convert the ndarray mask to tensor to be interpreted as boolean mask (instead of integer mask)
        data_2d = {seq: preprocessed2d[dataset.index.seq == seq] for seq in sequences}
        data_3d = {seq: preprocessed3d[dataset.index.seq == seq] for seq in sequences}
        data_3d_mm = {seq: post_process3d(preprocessed3d[dataset.index.seq == seq]) for seq in sequences}

        return LogAllMillimeterError(data_2d, data_3d, data_3d_mm, dataset.pose3d_jointset, loss, post_process3d, model, prefix)

    def __init__(self, data_2d, data_3d, data_3d_mm, joint_set, loss, post_process3d=None, model=None, prefix='val'):
        """
        Parameters:
            data_2d: preprocessed input for neural network, grouped by sequence
            data_3d: preprocessed target for neural network, grouped by sequence
            data_3d_mm: grount-truth 3D positions in mm, grouped by sequence
            joint_set: JointSet object describing the (output) joint orders
            post_process3d: 3D unnormalisation function, input is (nPoses, nJoints*3), output is (nPoses, nJoints, 3)
        """
        assert _sample_value(data_3d).shape[1] in [3 * joint_set.NUM_JOINTS, 3 * (joint_set.NUM_JOINTS - 1)], \
            "Unexpected shape: " + str(_sample_value(data_3d).shape)

        super().__init__(data_3d_mm, joint_set, post_process3d=post_process3d, prefix=prefix)

        self.model = model

        # Make sure data_2d values are torch DataLoaders
        if isinstance(_sample_value(data_2d), np.ndarray):
            self.data_2d = {k: DataLoader(TensorDataset(torch.from_numpy(v).cuda()), 256) for k, v in data_2d.items()}
        elif isinstance(_sample_value(data_2d), torch.Tensor):
            self.data_2d = {k: DataLoader(TensorDataset(v.cuda()), 256) for k, v in data_2d.items()}
        else:
            self.data_2d = data_2d

        # turn data3d into numpy arrays if tensor
        self.data_3d = {}
        for seq in data_3d.keys():
            arr = data_3d[seq]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()

            assert isinstance(arr, np.ndarray), "Unexpected type: " + str(type(arr))
            self.data_3d[seq] = arr

        # The loss without the mean
        if loss == 'l1' or loss == 'l1_nan':
            self.loss = lambda p, t: np.abs(p - t)
        elif loss == 'l2':
            self.loss = lambda p, t: np.square(p - t)
        else:
            raise Exception('unimplemented loss:' + str(loss))

    def pred_and_calc_loss(self, model):
        preds = {}
        losses = {}
        for seq in self.sequences:
            # calculate neural net predictions for this group
            pred = torch_predict(model if self.model is None else self.model, self.data_2d[seq])

            losses[seq] = self.loss(pred, self.data_3d[seq])
            preds[seq] = self.post_process3d(pred)

        return losses, preds
