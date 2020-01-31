import numpy as np


def depth_from_coords(depth_map, data, width, height):
    """
    Reads out values from a depth map at the given coordinates.

    Parameters:
        depth_map: input depth map, (nFrames, h, w)
        data: array of (nFrames, nPoints, [x, y])
        width, height: width and height of the original image (the scale the `data` parameter is using)

    Returns:
        ndarray(nFrames, nPoints), the depth at the given point
    """
    assert depth_map.ndim == 3
    assert data.ndim == 3 and data.shape[2] == 2, data.shape

    # Create an index map for each points
    inds = np.tile(np.arange(len(data)), (data.shape[1], 1)).T  # Simply the frame indices, duplicated in two columns
    points = np.concatenate([np.expand_dims(inds, 2), data], axis=2)  # result is (nFrames, nPoses, [frameInd, x, y])

    # Scale to depth map size
    points[:, :, 1] = points[:, :, 1] / width * depth_map.shape[2]
    points[:, :, 2] = points[:, :, 2] / height * depth_map.shape[1]
    points = np.around(points).astype('int32')

    # If the joint is out of the screen it can have invalid values
    points[:, :, 1] = np.clip(points[:, :, 1], 0, depth_map.shape[2] - 1)
    points[:, :, 2] = np.clip(points[:, :, 2], 0, depth_map.shape[1] - 1)

    depth = np.empty(data.shape[:2])
    for i in range(depth.shape[1]):
        depth[:, i] = depth_map[points[:, i, 0], points[:, i, 2], points[:, i, 1]]

    return depth


def project_points(calib, points3d):
    """
    Projects 3D points using a calibration matrix.

    Parameters:
        points3d: ndarray of shape (nPoints, 3)
    """
    assert points3d.ndim == 2 and points3d.shape[1] == 3

    p = np.empty((len(points3d), 2))
    p[:, 0] = points3d[:, 0] / points3d[:, 2] * calib[0, 0] + calib[0, 2]
    p[:, 1] = points3d[:, 1] / points3d[:, 2] * calib[1, 1] + calib[1, 2]

    return p


def calibration_matrix(points2d, points3d):
    """
    Calculates camera calibration matrix (no distortion) from 3D points and their projection.
    Only works if all points are away from the camera, eg all z coordinates>0.

    Returns:
        calib, reprojection error, x residuals, y residuals, x singular values, y singular values
    """
    assert points2d.ndim == 2 and points2d.shape[1] == 2
    assert points3d.ndim == 2 and points3d.shape[1] == 3

    A = np.column_stack([points3d[:, 0] / points3d[:, 2], np.ones(len(points3d))])
    px, resx, _, sx = np.linalg.lstsq(A, points2d[:, 0], rcond=None)

    A = np.column_stack([points3d[:, 1] / points3d[:, 2], np.ones(len(points3d))])
    py, resy, _, sy = np.linalg.lstsq(A, points2d[:, 1], rcond=None)

    calib = np.eye(3)
    calib[0, 0] = px[0]
    calib[1, 1] = py[0]
    calib[0, 2] = px[1]
    calib[1, 2] = py[1]

    # Calculate mean reprojection error
    # p = np.empty((len(points3d), 2))
    # p[:, 0] = points3d[:, 0] / points3d[:, 2] * calib[0, 0] + calib[0, 2]
    # p[:, 1] = points3d[:, 1] / points3d[:, 2] * calib[1, 1] + calib[1, 2]
    p = project_points(calib, points3d)
    reproj = np.mean(np.abs(points2d - p))

    return calib, reproj, resx, resy, sx, sy

