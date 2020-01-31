"""
Evaluates a (not end2end) model on MuPo-TS
"""
import argparse

import numpy as np
from torchvision.transforms import Compose

from databases import mupots_3d
from databases.datasets import FilteredSinglePersonMuPoTsDataset
from databases.joint_sets import MuPoTSJoints, CocoExJoints
from training.callbacks import LogAllMillimeterError
from util.experiments import load_model, load_transforms
from util.pose import combine_pose_and_trans


def unstack_poses(dataset, logger):
    """ Converts output of the logger to dict of list of ndarrays. """
    COCO_TO_MUPOTS = []
    for i in range(MuPoTSJoints.NUM_JOINTS):
        try:
            COCO_TO_MUPOTS.append(CocoExJoints().index_of(MuPoTSJoints.NAMES[i]))
        except:
            COCO_TO_MUPOTS.append(-1)
    COCO_TO_MUPOTS = np.array(COCO_TO_MUPOTS)
    assert np.all(COCO_TO_MUPOTS[1:14] >= 0)

    pred_2d = {}
    pred_3d = {}
    for seq in range(1, 21):
        gt = mupots_3d.load_gt_annotations(seq)
        gt_len = len(gt['annot2'])

        pred_2d[seq] = []
        pred_3d[seq] = []
        for i in range(gt_len):
            seq_inds = (dataset.index.seq == seq)
            frame_inds = (dataset.index.frame == i)
            pred_2d[seq].append(dataset.poses2d[seq_inds & frame_inds, :, :2][:, COCO_TO_MUPOTS])
            pred_3d[seq].append(logger.preds[seq][frame_inds[seq_inds]])

    return pred_2d, pred_3d


def main(model_name):
    config, m = load_model(model_name)

    test_set = FilteredSinglePersonMuPoTsDataset('hrnet', config.get('pose3d_scaling', 'normal'))

    transforms = load_transforms(model_name, config, test_set)
    test_set.transform = Compose(transforms)

    mean3d = transforms[1].normalizer.mean
    std3d = transforms[1].normalizer.std

    def post_process_func(x):
        return combine_pose_and_trans(x, std3d, mean3d, MuPoTSJoints(), 'hip')

    logger = LogAllMillimeterError.eval_model(m, test_set, config['pose_net']['loss'], post_process_func)

    pred_2d, pred_3d = unstack_poses(test_set, logger)
    print("\n%13s  R-PCK  R-AUC  A-PCK  A-AUC" % '')
    for detected_only in [True, False]:
        print("%13s: " % ("detected only" if detected_only else "all poses"), end='')
        for relative in [True, False]:
            pcks, aucs = mupots_3d.eval_poses(detected_only, relative, 'annot3' if model_name == 'unnormalized' else 'univ_annot3', pred_2d,
                                              pred_3d)
            pck = np.mean(list(pcks.values()))
            auc = np.mean(list(aucs.values()))

            print(" %4.1f   %4.1f  " % (pck, auc), end='')
        print()
    print()
    print("Poses found: %d / 20899 = %f" % (len(test_set), len(test_set) / 20899.))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help="Name of the model (either 'normalized' or 'unnormalized')")
    args = parser.parse_args()

    main(args.model_name)
