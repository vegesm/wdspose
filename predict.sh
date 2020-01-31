#!/usr/bin/env bash

# Runs the model on a folder of images
# Usage:
#   predict.sh <path/to/image_folder> <metadata.csv>
#
# The metadata.csv files contains the focal length and central points of each image (see example/metadata.csv).
# You have to set up the path to Detectron and Hrnet below. You might also want to activate any virtualenv necessary
# for running the Detectron/HR-net below
DETECTRON_FOLDER=~/pkgs/detectron/
HRNET_FOLDER=~/pkgs/deep-hrnet

IMG_FOLDER=$1
METADATA_FILE=$2

TMP_FOLDER=tmp
mkdir -p $TMP_FOLDER

# Mask-rcnn
bash -c "source activate detectron && DETECTRON_PATH=$DETECTRON_FOLDER python3 scripts/maskrcnn.py $IMG_FOLDER $TMP_FOLDER/bboxes"

source ~/pytorch1/bin/activate
HRNET_PATH=$HRNET_FOLDER python3 scripts/hrnet.py --imgs $IMG_FOLDER --bbox $TMP_FOLDER/bboxes --out $TMP_FOLDER/keypoints.json  \
--cfg $HRNET_FOLDER/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
deactivate

python3 scripts/megadepth.py  $IMG_FOLDER $TMP_FOLDER/megadepth
python3 scripts/predict.py $IMG_FOLDER $METADATA_FILE $TMP_FOLDER/keypoints.json $TMP_FOLDER/megadepth results.pkl -v
