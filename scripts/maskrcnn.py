import os
import argparse
import sys

DETECTRON_PATH = os.environ['DETECTRON_PATH']
sys.path.append(DETECTRON_PATH)

import cv2
from util.misc import save


from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file, merge_cfg_from_list
from detectron.utils.io import cache_url
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils

CONFIG_PATH = DETECTRON_PATH + '/configs/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml'
MODEl_URL = 'https://dl.fbaipublicfiles.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
GPU_ID = 0


def load_model():
    c2_utils.import_detectron_ops()

    # OpenCL may be enabled by default in OpenCV3; disable it because it's not
    # thread safe and causes unwanted GPU memory allocations.
    cv2.ocl.setUseOpenCL(False)

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=4'])
    merge_cfg_from_file(CONFIG_PATH)
    cfg.NUM_GPUS = 1

    print("Loading weights")
    weights = cache_url(MODEl_URL, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(weights, gpu_id=GPU_ID)
    print("Model loaded")

    return model


def predict(in_folder, out_folder):
    model = load_model()

    print("Running model...")
    for file in os.listdir(in_folder):
        img = cv2.imread(os.path.join(in_folder, file))
        with c2_utils.NamedCudaScope(GPU_ID):
            cls_boxes, _, _ = infer_engine.im_detect_all(model, img, None)

        save(os.path.join(out_folder, "%s.pkl" % file), cls_boxes[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder', help="Folder containing the images")
    parser.add_argument('out_folder', help="Results are saved here")
    args = parser.parse_args()

    predict(args.img_folder, args.out_folder)
