import torch
from torch import nn

from networks.torch_martinez import martinez_net, default_config


class CombinedModel(nn.Module):
    """
    Model that has a trunk to estimate 3d pose from 2d (pose_net) and
    a branch for weak training data predicting jointwise depth values from the 3D pose.
    """

    def __init__(self, input_size, output_size, weak_output_size,
                 posenet_params=None, weak_decoder_params=None):
        super(CombinedModel, self).__init__()

        conf = default_config()
        if posenet_params is not None:
            conf.update_values(posenet_params)
        _, pose_net = martinez_net(conf, input_size, output_size)
        self.pose_net = pose_net

        conf = default_config()
        if weak_decoder_params is not None:
            conf.update_values(weak_decoder_params)
        _, decoder = martinez_net(conf, output_size, weak_output_size)
        self.decoder = decoder

        self.weak_output_size = (weak_output_size,)

    def forward(self, pose2d, has_pose_annot):
        """
        pose2d: batch of 2D poses, (nPoses, input_size)
        has_pose_annot: true if the sample has pose annotation and not only weak depth, (nPoses, 1)
        """

        pose3d = self.pose_net(pose2d)
        depths = torch.zeros((pose3d.shape[0],) + self.weak_output_size).to(pose2d.device)

        if not has_pose_annot.all():
            weak_pose3d = pose3d[~has_pose_annot]
            depths[~has_pose_annot] = self.decoder(weak_pose3d)

        return pose3d, depths
