"""Functions to visualize human poses"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 0-right, 1-left, 2-center
from databases.joint_sets import MuPoTSJoints
from util.misc import assert_shape


def get_3d_axes(*subplot):
    """
    Creates a 3D Matplotlib axis. The arguments are the same as of the `subplot` function of Matplotlib.
    """
    if len(subplot) > 0:
        return plt.subplot(*subplot, projection='3d')
    else:
        return plt.subplot(1, 1, 1, projection='3d')


# alias to get_3d_axes
subplot = get_3d_axes


def add3Dpose(pose, ax, joint_set, lcolor="#d82f2f", rcolor="#e77d3c", ccolor="#2e78d8", line_width=2):
    """
    Plots a 3d skeleton on ``ax``.

    Parameters:
      pose: (nJoints, 3) ndarray. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      ccolor: color for the center of the body
      line_width: width of the limbs on the plot
    """
    assert pose.shape == (joint_set.NUM_JOINTS, 3), pose.shape

    # 0-right, 1-left, 2-center
    side = np.array(joint_set.SIDEDNESS, dtype='int32')
    colors = [rcolor, lcolor, ccolor]

    # Make connection matrix
    for i, limb in enumerate(joint_set.LIMBGRAPH):
        x = pose[limb, 0]
        y = pose[limb, 1]
        z = pose[limb, 2]

        # In Matplotlib z coordinate is the vertical axis and y is depth,
        # so we have to switch them
        y, z = z, y

        ax.plot(x, y, z, lw=line_width, c=colors[side[i]])


def show3Dpose(poses, joint_set, ax=None, lcolor="#d82f2f", rcolor="#e77d3c", ccolor="#2e78d8", add_labels=False, show_numbers=False,
               set_axis=True, hide_panes=False, radius=None,
               linewidth=2, invert_vertical=False):
    """
    Visualize a 3d skeleton. By default, left side is red, right is orange.

    Parameters:
      poses: ([nPoses], nJoints, 3) ndarray. The poses to plot. The joints must be in x,y,z (horizontal, vertical, depth) order.
      joint_set: JointSet object describing the joint orders.
      ax: matplotlib 3d axis to draw on, if None a new one is created
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      ccolor: color for the center of the body (spine, head, neck)
      add_labels: whether to add coordinates to the plot
      show_numbers: whether to show axis labels or not
      set_axis: if True, the limits of the plot axes are automatically set based on `poses`.
      radius: is set_axis is true, the half of the length of the viewport cube. If None, it is automatically inferred from data
      linewidth: width of the limbs on the plot
      invert_vertical: if true, the vertical axis grows downwards.
    """
    if poses.ndim == 2:
        poses = np.expand_dims(poses, 0)

    assert poses.shape[1:] == (joint_set.NUM_JOINTS, 3), poses.shape

    if ax is None:
        ax = get_3d_axes()

    for pose in poses:
        add3Dpose(pose, ax, joint_set, lcolor, rcolor, ccolor, linewidth)

    if set_axis:
        # space around the subject, automatically detect if it is in meters or mms
        if radius is None:
            radius = 1.500 if np.max(poses[0, :, 0]) - np.min(poses[0, :, 0]) < 5 else 1500
        xroot, yroot, zroot = np.nanmean(poses[:, joint_set.index_of("hip"), :], axis=0)
        ax.set_xlim3d([-radius + xroot, radius + xroot])
        ax.set_ylim3d([-radius + zroot, radius + zroot])
        ax.set_zlim3d([-radius + yroot, radius + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")

    # Get rid of the ticks and tick labels
    if not show_numbers:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

    if hide_panes:
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        ax.w_zaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)

    if invert_vertical:
        ax.invert_zaxis()

    ax.set_aspect('equal')


def show2Dpose(pose, joint_set, ax=None, lcolor="#d82f2f", rcolor="#e77d3c", ccolor="#2e78d8", add_labels=False, show_numbers=False,
               line_width=2, radius=500):
    """
    Visualize a 2d skeleton.

    Parameters:
      pose: (nJoints, 2) ndarray, the pose to draw.
      joint_set: The JointSet object that describes the joint order.
      ax: matplotlib axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      ccolor: color for the middle of the body
      add_labels: whether to add coordinate labels
      show_numbers: whether to show axis labels or not
      line_width: width of the plotted lines in pixels
      radius: half-width of the output image
    """
    assert pose.shape == (joint_set.NUM_JOINTS, 2), "Unexpected shape for pose:" + str(pose.shape)

    if ax is None:
        ax = plt.gca()

    # 0-right, 1-left, 2-center
    side = np.array(joint_set.SIDEDNESS, dtype='int32')
    colors = [rcolor, lcolor, ccolor]

    # Make connection matrix
    for i, limb in enumerate(joint_set.LIMBGRAPH):
        x = pose[limb, 0]
        y = pose[limb, 1]
        ax.plot(x, y, lw=line_width, c=colors[side[i]])

    if not show_numbers:
        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Get rid of tick labels
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

    # set space around the subject
    hip_ind = joint_set.index_of('hip')
    xroot, yroot = pose[hip_ind, 0], pose[hip_ind, 1]
    ax.set_xlim([-radius + xroot, radius + xroot])
    ax.set_ylim([-radius + yroot, radius + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.set_aspect('equal')
    ax.invert_yaxis()


def show_frame_poses(frame, poses2d, poses3d, joint_set2d, joint_set3d=None):
    """
    Shows a widget with the frame image, 2D and 3D poses.
    The 3D joint set is optional, if not specified `joint_set2d` is used.
    """

    if joint_set3d is None:
        joint_set3d = joint_set2d

    plt.subplot(1, 3, 1)
    plt.imshow(frame)

    plt.subplot(1, 3, 2)
    show2Dpose(poses2d, joint_set2d)

    ax = get_3d_axes(1, 3, 3)
    show3Dpose(poses3d, joint_set3d, ax=ax, invert_vertical=True)
    plt.tight_layout()


def gray2rgb(im, cmap='gray'):
    """ Convert a grayscale image to RGB using the `cmap` color map."""
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, clip=True, cmap='gray'):
    # convert to disparity, sky can have large values
    depth = 1. / (depth + 1e-6)
    if normalizer is not None:
        depth = depth / normalizer
    else:
        depth = depth / (np.percentile(depth, pc) + 1e-6)

    if clip:
        depth = np.clip(depth, 0, 1)

    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1 - crop_percent))
    depth = depth[:keep_H]
    return depth


def depth2rgb(data, normalizer):
    """ Converts a depth image to RGB. Normalizer is the maximum depth, values larger than this are clipped. Useful for the sky."""
    data = data.astype('float32') / normalizer
    data = np.clip(data, 0, 1)
    data = gray2rgb(data, 'viridis') * 255
    data = data.astype('uint8')

    return data


def draw_points(img, points, radius, color, thickness=-1):
    """
    Draws a set of points on img, also does rounding if necessary
    """
    assert points.ndim == 2 and points.shape[1] == 2

    points = np.around(points).astype('int32')
    for p in points:
        cv2.circle(img, (p[0], p[1]), radius, color, thickness)

    return img


def draw2Dpose(frame, pose, joint_set, lcolor=(216, 47, 47), rcolor=(231, 125, 60), ccolor=(46, 120, 216), line_width=2,
               score_threshold=0.4):
    """
    Draws a 2d skeleton on an image. Optionally filters points and edges by scores

    Parameters:
        frame: the image to draw on
        pose: (nJoints, [x,y,score]) vector, score is optional.
        joint_set: JointSet object describing the joint order.
        lcolor: color for left part of the body
        rcolor: color for right part of the body
        ccolor: color for the middle of the body
        line_width: width of the plotted lines in pixels
        score_threshold: joints with scores lower than this value are not drawn. Only applicable if ``pose`` contains scores.
    """
    assert pose.shape == (joint_set.NUM_JOINTS, 3) or pose.shape == (joint_set.NUM_JOINTS, 2), \
        "pose must have a shape of (%d, 2|3), it has %s instead" % (joint_set.NUM_JOINTS, str(pose.shape))
    coords = np.around(pose[:, :2]).astype('int32')

    if pose.shape[-1] == 3:
        scores = pose[:, 2]
    else:
        scores = np.ones(pose.shape[:-1])

    # 0-right, 1-left, 2-center
    side = np.array(joint_set.SIDEDNESS, dtype='int32')
    colors = [rcolor, lcolor, ccolor]

    # Draw joints
    for i, p in enumerate(coords):
        if scores[i] > score_threshold:
            cv2.circle(frame, (p[0], p[1]), 2 * line_width, (255, 0, 0), -1)

    # Draw limbs
    for i, limb in enumerate(joint_set.LIMBGRAPH):
        x = coords[limb, 0]
        y = coords[limb, 1]

        # Draw limb if both points were found
        if np.all(scores[list(limb)] > score_threshold):
            cv2.line(frame, (x[0], y[0]), (x[1], y[1]), colors[side[i]], line_width, cv2.LINE_AA)
        # ax.plot(x, y, lw=line_width, c=)


def create_prediction_viz_frames(gt, pred, valid, out_path, draw_func=None):
    """
    Creates frames showing the GT and predicted poses.

    Parameters:
        gt: ndarray(nFrames, nPoses, 17, 3) - ground-truth 3D coordinates using MuPoTS joints
        pred: ndarray(nFrames, nPoses, 17, 3) - predicted 3D coordinates using MuPoTS joints
        valid: ndarray(nFrames, nPoses) - true if the given pose was detected
        out_path: path to the frame, must have a %d format specifier for the image index.
        draw_func: optional function that draws the 3D plot It receives the gt, predicted joints, valid joint index and a 3D matplotlib axes.
                    if None, it simply draws gt skeletons in dark, predicted skeletons light.
    """
    assert_shape(gt, (None, None, 17, 3))
    assert_shape(pred, (None, None, 17, 3))

    assert gt.shape[:2] == valid.shape
    assert gt.shape[:2] == pred.shape[:2]

    plt.ioff()

    RADIUS = 1500
    plt.clf()
    ax = get_3d_axes()
    xroot, yroot, zroot = np.mean(gt[0][:, 14, :], axis=0)
    bottom = np.max(gt[0][:, :, 1])

    def default_drawer(gts, preds, valid, ax):
        show3Dpose(gts, MuPoTSJoints(), ax=ax, invert_vertical=True, show_numbers=False, lcolor="#911f1f",
                   rcolor="#874924", ccolor="#1b4882")
        for p in preds[valid]:
            add3Dpose(p, ax, MuPoTSJoints())

        ax.set_xlim3d([-RADIUS - 400 + xroot, RADIUS + xroot + 600])
        ax.set_ylim3d([-RADIUS + zroot - 200, RADIUS + zroot + 100])
        ax.set_zlim3d([bottom + 10, bottom - 2500])

    if draw_func is None:
        draw_func = default_drawer

    for i in range(len(gt)):
        plt.cla()
        draw_func(gt[i], pred[i], valid[i], ax)
        plt.savefig(out_path % i)

    plt.ion()


def generate_rotating_pose(out_file, pose, joint_set, initial_func=None):
    """
    Creates a gif that rotates the camera around `pose`. You might want to call ``plt.ioff()`` before
    using this function.

    Parameters:
        out_file: output file name
        pose: ndarray(nJoints,3), coordinates are in x, y, z order. Y grows downwards.
        joint_set: joint order of `pose`
        initial_func: if not None, a function that generates the initial plot that is going to be rotated. If provided, pose and
                        joint_set must be NaN
    """
    if initial_func is not None:
        assert pose is None, "if initial_func i set, pose must be None"
        assert joint_set is None, "if initial_func i set, joint_set must be None"

    start_azim = -190
    end_azim = -0
    azim_steps = 30

    start_elev = 5
    end_elev = 70
    elev_steps = 15

    total_steps = azim_steps * 2 + elev_steps * 2

    def update(i):
        middle_azim = (start_azim + end_azim) / 2
        middle_elev = (start_elev + end_elev) / 2

        if i < azim_steps / 2:
            ax.view_init(azim=middle_azim + (start_azim - middle_azim) * i / azim_steps * 2, elev=middle_elev)
        if azim_steps / 2 <= i < azim_steps * 1.5:
            ax.view_init(azim=start_azim + (end_azim - start_azim) * (i - azim_steps / 2) / azim_steps, elev=middle_elev)
        elif azim_steps * 1.5 <= i < azim_steps * 2:
            ax.view_init(azim=end_azim + (middle_azim - end_azim) * (i - azim_steps * 1.5) / azim_steps * 2, elev=middle_elev)

        elif azim_steps * 2 <= i < azim_steps * 2 + elev_steps / 2:
            ax.view_init(azim=middle_azim, elev=middle_elev + (start_elev - middle_elev) * (i - azim_steps * 2) / elev_steps * 2)
        elif azim_steps * 2 + elev_steps / 2 <= i < azim_steps * 2 + elev_steps * 1.5:
            ax.view_init(azim=middle_azim, elev=start_elev + (end_elev - start_elev) *
                                                (i - azim_steps * 2 - elev_steps / 2) / elev_steps)
        elif azim_steps * 2 + elev_steps * 1.5 <= i:
            ax.view_init(azim=middle_azim, elev=end_elev + (middle_elev - end_elev) *
                                                (i - azim_steps * 2 - elev_steps * 1.5) / elev_steps * 2)

    def first_frame():
        show3Dpose(pose, joint_set, ax, invert_vertical=True, linewidth=1, set_axis=False, hide_panes=False)
        RADIUS = 900
        xroot, yroot, zroot = np.mean(pose[:, joint_set.index_of('hip'), :], axis=0)
        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        ax.set_ylim3d([-RADIUS + zroot, RADIUS + zroot])
        ax.set_zlim3d([-RADIUS + yroot, RADIUS + yroot])
        ax.invert_zaxis()
        ax.set_aspect('equal')

    if initial_func is None:
        initial_func = first_frame

    ax = get_3d_axes()
    plt.tight_layout()
    fps = 10
    anim = FuncAnimation(plt.gcf(), update, frames=total_steps,
                         interval=1000 / fps, init_func=initial_func, repeat=False)

    writer = ImageMagickWriter(fps=fps)
    anim.save(out_file, writer=writer)


class Arrow3D(FancyArrowPatch):
    """
    Draws an arrow on a 3D plot.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
