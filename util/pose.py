import numpy as np

from databases.joint_sets import CocoExJoints
from util.misc import assert_shape


def remove_root(data, root_ind):
    """
    Removes a joint from a dataset by moving it to the origin and removing it from the array.

    :param data: (..., nJoints, 2|3) array
    :param root_ind: index of the joint to be removed
    :return: (..., nJoints-1, 2|3) array
    """
    assert data.ndim >= 2 and data.shape[-1] in (2, 3)

    roots = data[..., [root_ind], :]  # (..., 1, [2|3])
    data = data - roots
    data = np.delete(data, root_ind, axis=-2)

    return data


def remove_root_keepscore(data, root_ind):
    """
    Removes a joint from a 2D dataset by moving to the origin and removing it from the array.
    The difference to remove_root is that the third column stores the confidence score and it is
    not changed.

    :param data: (nPoses, nJoints, 3[x,y,score]) array
    :param root_ind: index of the joint to be removed
    :return: (nPoses, nJoints-1, 3[x,y,score]) array
    """
    assert data.ndim >= 3 and data.shape[-1] == 3, data.shape

    roots = data[..., [root_ind], :2]  # ndarray(...,1,2)
    # roots = roots.reshape((len(roots), 1, 2))
    data[..., :2] = data[..., :2] - roots
    data = np.delete(data, root_ind, axis=-2)

    return data


def combine_pose_and_trans(data3d, std3d, mean3d, joint_set, root_name):
    """
    3D result postprocess: unnormalizes data3d and reconstructs the absolute pose from relative + absolute split.

    Parameters:
        data3d: output of the PyTorch model, ndarray(nPoses, 3*nJoints), in the format created by preprocess3d
        std3d: normalization standard deviations
        mean3d: normalization means
        root_name: name of the root joint

    Returns:
        ndarray(nPoses, nJoints, 3)
    """
    assert_shape(data3d, (None, joint_set.NUM_JOINTS * 3))

    data3d = data3d * std3d + mean3d
    root = data3d[:, -3:]
    rel_pose = data3d[:, :-3].reshape((len(data3d), joint_set.NUM_JOINTS - 1, 3))

    root[:, 2] = np.exp(root[:, 2])

    rel_pose += root[:, np.newaxis, :]

    result = np.zeros((len(data3d), joint_set.NUM_JOINTS, 3), dtype='float32')
    root_ind = joint_set.index_of(root_name)
    result[:, :root_ind, :] = rel_pose[:, :root_ind, :]
    result[:, root_ind, :] = root
    result[:, root_ind + 1:, :] = rel_pose[:, root_ind:, :]

    return result


def harmonic_mean(a, b, eps=1e-6):
    return 2 / (1 / (a + eps) + 1 / (b + eps))


def _combine(data, target, a, b):
    """
    Modifies data by combining (taking average) joints at index a and b at position target.
    """
    data[:, target, :2] = (data[:, a, :2] + data[:, b, :2]) / 2
    data[:, target, 2] = harmonic_mean(data[:, a, 2], data[:, b, 2])


def extend_hrnet_raw(raw):
    assert_shape(raw, (None, 17, 3))
    js = CocoExJoints()

    result = np.zeros((len(raw), 19, 3), dtype='float32')
    result[:, :17, :] = raw
    _combine(result, js.index_of('hip'), js.index_of('left_hip'), js.index_of('right_hip'))
    _combine(result, js.index_of('neck'), js.index_of('left_shoulder'), js.index_of('right_shoulder'))

    return result
