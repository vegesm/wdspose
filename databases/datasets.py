import numpy as np
from torch.utils.data import Dataset

from databases import mupots_3d
from databases.joint_sets import MuPoTSJoints, CocoExJoints, OpenPoseJoints


class FilteredSinglePersonMuPoTsDataset(Dataset):
    def __init__(self, pose2d_type, pose3d_scaling):
        """
        Loads MuPoTS dataset but only those images where at least one person was detected. Each person on a frame
        is loaded separately.
        """
        assert pose3d_scaling in ['univ', 'normal']

        self.pose2d_jointset = FilteredSinglePersonMuPoTsDataset.get_jointset(pose2d_type)
        self.pose3d_jointset = MuPoTSJoints()

        poses2d = []
        poses3d = []
        pred_cdepths = []
        index = []
        for seq in range(1, 21):
            depth_width = 512
            depth_height = 512 if seq <= 5 else 288

            gt = mupots_3d.load_gt_annotations(seq)
            op = mupots_3d.load_2d_predictions(seq, pose2d_type)

            pose2d = op['pose']
            pose3d = gt['annot3' if pose3d_scaling == 'normal' else 'univ_annot3']

            depth = mupots_3d.load_jointwise_depth(seq)

            good_poses = gt['isValidFrame'].squeeze()
            good_poses = np.logical_and(good_poses, op['valid_pose'])

            orig_frame = np.tile(np.arange(len(good_poses)).reshape((-1, 1)), (1, good_poses.shape[1]))
            orig_pose = np.tile(np.arange(good_poses.shape[1]).reshape((1, -1)), (good_poses.shape[0], 1))

            assert pose2d.shape[:2] == good_poses.shape  # (nFrames, nPeople)
            assert pose3d.shape[:2] == good_poses.shape
            assert depth.shape[:2] == good_poses.shape
            assert orig_frame.shape == good_poses.shape
            assert orig_pose.shape == good_poses.shape
            assert pose2d.shape[2:] == (self.pose2d_jointset.NUM_JOINTS, 3)
            assert pose3d.shape[2:] == (17, 3)
            assert good_poses.ndim == 2

            # Keep only those poses where good_poses is True
            pose2d = pose2d[good_poses]
            pose3d = pose3d[good_poses]
            orig_frame = orig_frame[good_poses]
            orig_pose = orig_pose[good_poses]
            depth = depth[good_poses]

            index.extend([(seq, orig_frame[i], orig_pose[i], depth_width, depth_height)
                          for i in range(len(orig_frame))])

            assert len(pose2d) == len(pose3d)

            poses2d.append(pose2d)
            poses3d.append(pose3d)
            pred_cdepths.append(depth)

        self.poses2d = np.concatenate(poses2d).astype('float32')
        self.poses3d = np.concatenate(poses3d).astype('float32')
        self.pred_cdepths = np.concatenate(pred_cdepths).astype('float32')
        self.index = np.rec.array(index, dtype=[('seq', 'int32'), ('frame', 'int32'), ('pose', 'int32'),
                                                ('depth_width', 'int32'), ('depth_height', 'int32')])

        # Load calibration matrices
        N = len(self.poses2d)
        self.fx = np.zeros(N, dtype='float32')
        self.fy = np.zeros(N, dtype='float32')
        self.cx = np.zeros(N, dtype='float32')
        self.cy = np.zeros(N, dtype='float32')

        mupots_calibs = mupots_3d.get_calibration_matrices()
        for seq in range(1, 21):
            inds = (self.index.seq == seq)
            self.fx[inds] = mupots_calibs[seq][0, 0]
            self.fy[inds] = mupots_calibs[seq][1, 1]
            self.cx[inds] = mupots_calibs[seq][0, 2]
            self.cy[inds] = mupots_calibs[seq][1, 2]

        assert np.all(self.fx > 0), "Some fields were not filled"
        assert np.all(self.fy > 0), "Some fields were not filled"
        assert np.all(np.abs(self.cx) > 0), "Some fields were not filled"
        assert np.all(np.abs(self.cy) > 0), "Some fields were not filled"
        self.transform = None

    @staticmethod
    def get_jointset(pose2d_type):
        if pose2d_type == 'openpose':
            return OpenPoseJoints()
        elif pose2d_type == 'hrnet':
            return CocoExJoints()
        else:
            raise Exception("Unknown 2D pose type: " + pose2d_type)

    def __len__(self):
        return len(self.poses2d)

    def prepare_sample(self, ind):
        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind], 'pred_cdepth': self.pred_cdepths[ind],
                  'index': ind}

        return sample

    def __getitem__(self, idx):
        sample = self.prepare_sample(idx)

        if self.transform:
            sample = self.transform(sample)

        return sample
