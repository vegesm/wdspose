import argparse
import os

import cv2
import numpy as np
import torch

from networks import pytorch_DIW_scratch


def load_model():
    model = pytorch_DIW_scratch.pytorch_DIW_scratch
    model = torch.nn.parallel.DataParallel(model, device_ids=[0])
    model_parameters = torch.load('best_generalization_net_G.pth')
    model.load_state_dict(model_parameters)
    model.eval()

    return model


def recommended_size(img_shape):
    """
    Calculates the recommended size for a MegaDepth prediction.
    The width will be 512 pixels long, the height the nearest multiple of 32
    """
    new_width = 512
    new_height = img_shape[0] / img_shape[1] * 512
    new_height = round(new_height / 32) * 32
    return new_width, new_height


def predict_depth(frames, model, batch_size=16):
    """
    Generates depth maps for a video.

    :param frames: a list of frames to run the model on. Must be in RGB order.
    :param model: MegaDepth model
    :returns: array of [N,h,w] where N is the number of frames and h and w are the output height and width.
              These two can be different than the original input size.
    """
    assert frames.ndim == 4 and frames.shape[3] == 3, "channels should be the last dimension"

    preds = []
    batch_start = 0
    while batch_start < len(frames):
        batch_end = min(batch_start + batch_size, len(frames))
        batch = frames[batch_start:batch_end].astype('float32') / 255.0
        input_images = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2))).contiguous().float()

        input_images = input_images.cuda()
        pred_log_depth = model.forward(input_images)
        pred_depth = torch.exp(pred_log_depth)
        preds.append(pred_depth.data.cpu().numpy())

        # clean up GPU memory
        del pred_depth, pred_log_depth, input_images

        batch_start += batch_size

    preds = np.concatenate(preds, axis=0)
    assert preds.ndim == 4 and preds.shape[1] == 1
    preds = preds[:, 0, :, :]  # Remove channel dimension
    assert len(frames) == len(preds), (len(frames), len(preds))

    return preds


def predict(img_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    with torch.no_grad():
        model = load_model()
        for file in os.listdir(img_folder):
            img = cv2.imread(os.path.join(img_folder, file))
            width, height = recommended_size(img.shape)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, height))
            img = np.expand_dims(img, 0)

            depth = predict_depth(img, model, batch_size=1)
            np.save(os.path.join(out_folder, '%s.npy' % file), depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder', help="Folder containing the images")
    parser.add_argument('out_folder', help="Results are saved here")
    args = parser.parse_args()

    predict(args.img_folder, args.out_folder)
