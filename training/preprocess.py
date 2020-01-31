import numpy as np
import torch
from torch.utils.data import Dataset

from databases.joint_sets import Common14Joints, CocoExJoints
from util.misc import assert_shape, load
from util.pose import remove_root, remove_root_keepscore


def preprocess_2d(data, fx, cx, fy, cy, joint_set, root_name):
    """
    2D data preprocessing, performing the following:
        1. Keeps only COMMON14 joints
        2. Normalizes coordinates by multiplying with the inverse of the calibration matrix
        3. Converts numbers in a root-relative form
        4. Invisible joints are replaced by a single value
        5. Convert data into float

    :param data: (nPoses, 25, 3[x, y, scores]) - OpenPose detected coordinates
    :param fx: ndarray(nPoses) or float, horizontal focal length
    :param cx: ndarray(nPoses) or float, horizontal principal point
    :param fy: ndarray(nPoses) or float, vertical focal length
    :param cy: ndarray(nPoses) or float, horizontal principal point
    :param joint_set: the JointSet object describing the order of joints
    :param root_name: name of the root joint, must be a COMMON14 joint
    :return: ndarray(nPoses, 42), First 39 numbers are the non-root joints, last one is the root
    """
    assert_shape(data, ("*", None, joint_set.NUM_JOINTS, 3))
    assert not isinstance(fx, np.ndarray) or len(fx) == len(data)
    assert not isinstance(fy, np.ndarray) or len(fy) == len(data)

    if isinstance(fx, np.ndarray):
        N = len(data)
        shape = [1] * (data.ndim - 1)
        shape[0] = N
        fx = fx.reshape(shape)
        fy = fy.reshape(shape)
        cx = cx.reshape(shape)
        cy = cy.reshape(shape)

    data = data[..., joint_set.TO_COMMON14, :]

    data[..., :, 0] -= cx
    data[..., :, 1] -= cy
    data[..., :, 0] /= fx
    data[..., :, 1] /= fy

    root_ind = np.where(Common14Joints.NAMES == root_name)[0][0]
    root2d = data[..., root_ind, :].copy()
    data = remove_root_keepscore(data, root_ind)  # (nPoses, 13, 3), modifies data

    bad_frames = data[..., 2] < 0.1

    # replace joints having low scores with 1700/focus
    # this is to prevent leaking cx/cy
    if isinstance(fx, np.ndarray):
        fx = np.tile(fx, (1,) + data.shape[1:-1])
        fy = np.tile(fy, (1,) + data.shape[1:-1])
        data[bad_frames, 0] = -1700 / fx[bad_frames]
        data[bad_frames, 1] = -1700 / fy[bad_frames]
    else:
        data[bad_frames, 0] = -1700 / fx
        data[bad_frames, 1] = -1700 / fy

    # stack root next to the pose
    data = data.reshape(data.shape[:-2] + (-1,))  # (nPoses, 13*3)
    data = np.concatenate([data, root2d], axis=-1)  # (nPoses, 14*3)
    return data.astype('float32')


def preprocess_3d(data, add_root, joint_set, root_name):
    """
    3D preprocessing:
        1. Removes the root joint
        2. If add_root is True,  append the root joint at the end of the pose. The
           The logarithm of the z coordinate of the root is taken.
        3. Flattens the data.

    :param data: ndarray(nFrames, [nPoses], nJoints, 3[x, y, z]) 3D coordinates in MuPoTS order
    :param add_root: True if the absolute coordinates of the hip should be included in the output
    :param root_name: name of the root joint, must be a MuPoTS joint
    :return: ndarray(nPoses, 3*nJoints|3*(nJoints-1)), 3*nJoints if add_root is true otherwise 3*(nJoints-1)
    """
    assert_shape(data, ("*", joint_set.NUM_JOINTS, 3))

    root_ind = joint_set.index_of(root_name)
    root3d = data[..., root_ind, :].copy()
    root3d[..., 2] = np.log(root3d[..., 2])
    data = remove_root(data, root_ind)  # (nFrames, [nPoses], nJoints-1, 3)
    data = data.reshape(data.shape[:-2] + (-1,))  # (nFrames, [nPoses], (nJoints-1)*3)
    if add_root:
        data = np.concatenate([data, root3d], axis=-1)  # (nFrames, [nPoses], nJoints*3)

    return data.astype('float32')


class RemoveIndex(object):
    """
    Deletes the 'meta' field from the data item, useful for cleaning up for batching.
    """

    def __call__(self, sample):
        sample.pop('index', None)
        return sample


class ToTensor(object):
    """ Converts ndarrays in sample to pytorch tensors. Expects dicts as inputs. """

    def __call__(self, sample):
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v) for k, v in sample.items()}


class BaseNormalizer(object):
    """
    Baseclass for preprocessors that normalize a field.

    Subclasses must set the field_name field by themselves, outside the constructor.
    They must also have the constructor to accept a single 'None' argument, that does
    not preload the parameters.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def from_file(cls, path):
        state = load(path)
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state):
        """
        Path is a pkl file that contains mean and std.
        """
        instance = cls(None)
        instance.mean = state['mean']
        instance.std = state['std']

        return instance

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'field_name': self.field_name}

    def __call__(self, sample):
        sample[self.field_name] = (sample[self.field_name] - self.mean) / self.std
        return sample


class MeanNormalize2D(BaseNormalizer):
    """
    Normalizes the input 3D pose with mean and std.
    """

    def __init__(self, dataset):
        """
        Parameters:
            dataset:  either a numpy array containing the 3D poses or a PanopticSinglePersonDataset
        """
        self.field_name = 'pose2d'
        if dataset is None:
            # mean and std must be set manually later
            return

        if not isinstance(dataset, np.ndarray):
            dataset = dataset.poses2d

        assert isinstance(dataset, np.ndarray), "Expected dataset to be either a PanopticSinglePersonDataset or a numpy array, got:" + str(
            type(dataset))

        # data = dataset.reshape((len(dataset), -1))
        data = dataset.reshape((-1, dataset.shape[-1]))
        super().__init__(np.nanmean(data, axis=0), np.nanstd(data, axis=0))


class MeanNormalize3D(BaseNormalizer):
    """
    Normalizes the input 3D pose with mean and std.
    """

    def __init__(self, dataset):
        """
        Parameters:
            dataset:  either a numpy array containing the 3D poses or a PanopticSinglePersonDataset
        """
        self.field_name = 'pose3d'
        if dataset is None:
            # mean and std must be set manually later
            return

        assert isinstance(dataset, np.ndarray), "Expected dataset to be a numpy array"

        # data = dataset.reshape((len(dataset), -1))
        data = dataset.reshape((-1, dataset.shape[-1]))
        super().__init__(np.nanmean(data, axis=0), np.nanstd(data, axis=0))


class MeanChannelWiseNormalizeField(BaseNormalizer):
    """
    Normalizes a given field in the dataset.
    """

    def __init__(self, field_name, dataset):
        """
        dataset parameter is used for calculating mean and std. If None then the mean and
        std is not defiend an must be set manually.
        """
        self.field_name = field_name

        if dataset is None:
            return

        if isinstance(dataset, Dataset):
            prop_name = MeanChannelWiseNormalizeField.property_name_for(field_name)
            dataset = dataset.__getattribute__(prop_name)

        # data = dataset.reshape((len(dataset), -1))
        data = dataset.reshape((-1, dataset.shape[-1]))
        super().__init__(np.nanmean(data, axis=0), np.nanstd(data, axis=0))

    @classmethod
    def from_state(cls, state):
        """
        Path is a pkl file that contains mean and std.
        """
        instance = cls('', None)
        instance.mean = state['mean']
        instance.std = state['std']
        instance.field_name = state['field_name']

        return instance

    @staticmethod
    def property_name_for(field):
        if field == 'pose3d':
            return 'poses3d'
        if field == 'pose2d':
            return 'poses2d'
        if field == 'coord_depth':
            return 'coord_depths'
        if field == 'pred_cdepth':
            return 'pred_cdepths'


class SplitToRelativeAbsAndMeanNormalize3D(object):
    """
    Splits the 3D poses into relative+absolute and then normalizes it. It is uses the same
    preprocessing mechanics as the Depthpose paper did.
    """

    def __init__(self, dataset, normalizer=None, cache=False):
        """
        :param dataset: The full dataset, required if no normalizer is provided or ``cache`` is True.
        :param normalizer: the Normalizer object to be applied on the preprocessed data. If None,
                           the normalizer parameters are calculated from the dataset.
        :param cache: If True, preprocessed values are saved and not calculated every time during training.
                      Potenially speed up training.
        """
        if cache or normalizer is None:
            assert dataset is not None, "dataset must be defined if cache==true or no normalizer provided"
        self.cache = cache

        if dataset is not None:
            self.joint_set = dataset.pose3d_jointset
            preprocessed3d = preprocess_3d(dataset.poses3d, True, self.joint_set, 'hip')

            if normalizer is None:
                normalizer = MeanNormalize3D(preprocessed3d)

            if cache:
                self.preprocessed3d = (preprocessed3d - normalizer.mean) / normalizer.std

        assert isinstance(normalizer, MeanNormalize3D), \
            "Unexpected normalizer type: " + str(type(normalizer))
        self.normalizer = normalizer

    @classmethod
    def from_file(cls, path, dataset):
        state = load(path)
        return cls.from_state(state, dataset)

    @classmethod
    def from_state(cls, state, dataset):
        """
        Path is a pkl file that contains mean and std.
        """
        instance = cls(dataset, MeanNormalize3D.from_state(state), cache=False)
        if dataset is None:
            set_name = state['joint_set']
            if "<class '" in set_name:  # fixing incorrectly formatted type name
                set_name = set_name[set_name.rindex('.') + 1:-2]
            instance.joint_set = globals()[set_name]()

        return instance

    def state_dict(self):
        state = self.normalizer.state_dict()
        state['joint_set'] = str(type(self.joint_set))
        return state

    def __call__(self, sample):
        # Note: this algorithm makes iterating over all examples 9s slower, seems acceptable
        # pose3d = sample['pose3d']  # shape is (, nJoints*3)
        # preprocessed = preprocess_3d(pose3d.reshape((self.num_joints, 3)), True, PanopticJoints(), 'hip')
        if self.cache:
            preprocessed = self.preprocessed3d[sample['index']]
            sample['pose3d'] = preprocessed
        else:
            pose3d = sample['pose3d']  # shape is ([nPoses],nJoints, 3)
            preprocessed = preprocess_3d(pose3d, True, self.joint_set, 'hip')
            sample['pose3d'] = preprocessed
            sample = self.normalizer(sample)
        return sample


class DepthposeNormalize2D(object):
    """
    Normalizes the 2D pose using the technique in Depthpose.
    """

    def __init__(self, dataset, normalizer=None, cache=False):
        """
        :param dataset: The full dataset, required if no normalizer is provided or ``cache`` is True.
        :param normalizer: the Normalizer object to be applied on the preprocessed data. If None,
                           the normalizer parameters are calculated from the dataset.
        :param cache: If True, preprocessed values are saved and not calculated every time during training.
                      Potenially speed up training.
        """
        if cache or normalizer is None:
            assert dataset is not None, "dataset must be defined if cache==true or no normalizer provided"
        self.cache = cache

        if dataset is not None:
            preprocessed2d = preprocess_2d(dataset.poses2d.copy(), dataset.fx, dataset.cx, dataset.fy, dataset.cy,
                                           dataset.pose2d_jointset, 'hip')

            if normalizer is None:
                normalizer = MeanNormalize2D(preprocessed2d)

            if cache:
                self.preprocessed2d = (preprocessed2d - normalizer.mean) / normalizer.std

        self.normalizer = normalizer
        self.dataset = dataset
        assert isinstance(self.normalizer, MeanNormalize2D), \
            "Unexpected normalizer type: " + str(type(normalizer))

    @classmethod
    def from_file(cls, path, dataset):
        """
        Path is a pkl file that contains mean and std.
        """
        instance = cls(dataset, MeanNormalize2D.from_file(path), cache=False)

        return instance

    def state_dict(self):
        return self.normalizer.state_dict()

    def __call__(self, sample):
        if self.cache:
            sample['pose2d'] = self.preprocessed2d[sample['index']]
        else:
            pose2d = sample['pose2d']  # shape is ([nPoses],nJoints, 3)
            ind = sample['index']
            pose2d = np.expand_dims(pose2d, axis=0)
            preprocessed = preprocess_2d(pose2d.copy(), self.dataset.fx[ind], self.dataset.cx[ind],
                                         self.dataset.fy[ind], self.dataset.cy[ind],
                                         self.dataset.pose2d_jointset, 'hip')
            sample['pose2d'] = preprocessed[0]
            sample = self.normalizer(sample)
        return sample


class FuncAndNormalize(object):
    """
    A preprocessing step that applies a function on a single field with a normalisation step afterward.
    """

    def __init__(self, func, field, dataset, normalizer=None, cache=False):
        """
        The mean and std is learned from `dataset`, that must be a PanopticSinglePersonDataset.

        Parameters:
            func: a function that takes an ndarray of shape (nBatch,...) and produces (nBatch, ...)
                  The dimensions of the input and output arrays do not have to be the same, except for the first dimension.
            field: On which field the normalisation applied. Can be either 'pose2d' or 'pose3d'.
            dataset: PanopticSinglePersonDataset
            normalizer: The normalisation to be applied after `func`. If none, the normalisation params are determined automatically.
            cache: If true, the preprocessed values are cached to speed up training. But you must be sure
                   that other transformers don't modify the input values in the preprocessing pipeline!
        """
        if cache or normalizer is None:
            assert dataset is not None, "dataset must be defined if cache==true or no normalizer provided"
        assert field in ('pose3d', 'pose2d', 'coord_depth', 'pred_cdepth'), 'Unexpected field name: ' + field

        self.func = func
        self.field = field
        self.cache = cache

        if dataset is not None and (cache or normalizer is None):
            prop_name = MeanChannelWiseNormalizeField.property_name_for(field)
            data = dataset.__getattribute__(prop_name)
            preprocessed = func(data)

            if normalizer is None:
                normalizer = MeanChannelWiseNormalizeField(field, preprocessed)

            if cache:
                self.preprocessed = (preprocessed - normalizer.mean) / normalizer.std

        assert isinstance(normalizer, MeanChannelWiseNormalizeField), \
            "Unexpected normalizer type: " + str(type(normalizer))
        self.normalizer = normalizer

    @classmethod
    def from_file(cls, func, path, dataset):
        state = load(path)
        return cls.from_state(func, state, dataset)

    @classmethod
    def from_state(cls, func, state, dataset):
        """
        Path is a pkl file that contains mean and std.
        """
        # norm_class = FuncAndNormalize.normalizer_class_for(state['field'])
        instance = cls(func, state['field'], dataset, MeanChannelWiseNormalizeField.from_state(state), cache=False)

        return instance

    def state_dict(self):
        state = self.normalizer.state_dict()
        state['field'] = self.field
        return state

    def __call__(self, sample):
        if self.cache:
            preprocessed = self.preprocessed[sample['index']]
            sample[self.field] = preprocessed

        else:
            data = sample[self.field]
            preprocessed = self.func(np.expand_dims(data, axis=0))
            assert len(preprocessed) == 1
            preprocessed = preprocessed[0]
            sample[self.field] = preprocessed
            sample = self.normalizer(sample)
        return sample


def log_keep_hrnet_c14(data):
    return keep_hrnet_c14(np.log(data))


def concat_cdepth(sample):
    sample['pose2d'] = np.concatenate([sample['pose2d'], sample['pred_cdepth']], axis=-1)
    return sample


def keep_hrnet_c14(data):
    """
    Keeps only COMMON-14 joints from hrnet.
    data - ndarray(..., 19), along the last dimension, each slice corresponds to a joint In CocoEx joint order.
    """
    assert_shape(data, ('*', None, CocoExJoints.NUM_JOINTS))
    data = data[..., CocoExJoints.TO_COMMON14]

    return data


def decode_trfrm(transform_name, locals=None):
    """
    Converts a description of a transformation name into an actual transformation.

    Parameters:
        transform_name: Either the name of a Preprocess class, or as string in form 'FN(<field>, <func>)'.
                        In the second case a FuncAndNormalize class is created.
        locals: dict that contains the defined functions in the current scope. Useful for calling this function
                from outside preprocess.py where there are additional functions.
    """
    if transform_name.startswith('FN('):
        assert transform_name[-1] == ')', "Incorrectly formatted transform: " + transform_name
        transform_name = transform_name[3:-1]
        field, func = map(lambda x: x.strip(), transform_name.split(',', maxsplit=1))
        func = eval(func, globals(), locals)

        # wrapper to bind in func
        class FuncAndNormalizeWrapper(FuncAndNormalize):
            def __init__(self, dataset, normalizer=None, cache=False):
                super().__init__(func, field, dataset, normalizer, cache)

            @classmethod
            def from_file(cls, path, dataset):
                return FuncAndNormalize.from_file(func, path, dataset)

        return FuncAndNormalizeWrapper

    else:
        names = dict(globals())
        if locals is not None:
            names.update(locals)
        return names[transform_name]
