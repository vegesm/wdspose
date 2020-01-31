from networks.combined_model import CombinedModel
from networks.torch_martinez import martinez_net, default_config
import torch
from training.preprocess import decode_trfrm, concat_cdepth, RemoveIndex, ToTensor
from util.misc import load
import os

LOG_PATH = 'models'


def load_model(model_name):
    config = load(os.path.join(LOG_PATH, model_name, 'config.json'))

    # Input/output size calculation is hacky
    if config['model'] == 'depthpose':
        martinez_conf = default_config()
        martinez_conf.update_values(config['pose_net'])
        _, m = martinez_net(martinez_conf, 56, 17 * 3)

    elif config['model'] == 'depthpose_comb':
        m = CombinedModel(56, 17 * 3, 14, config['pose_net'], config['weak_decoder'])

    elif config['model'] == 'depthpose_offset':
        raise Exception("depthpose_offset only implemented in depthpos.ipynb")
    else:
        raise NotImplementedError("Unknown model: " + config['model'])

    m.cuda()
    m.load_state_dict(torch.load(os.path.join(LOG_PATH, model_name, 'model_params.pkl')))
    m.eval()
    if config['model'] in ('depthpose_comb', 'depthpose_offset'):
        m = m.pose_net
    return config, m


def load_transforms(model_name, config, dataset):
    preprocess_2d_path = os.path.join(LOG_PATH, model_name, 'preprocess_2d_params.pkl')
    preprocess_3d_path = os.path.join(LOG_PATH, model_name, 'preprocess_3d_params.pkl')
    preprocess_pcdepth_path = os.path.join(LOG_PATH, model_name, 'preprocess_pcdepth_params.pkl')

    return [decode_trfrm(config['preprocess_2d']).from_file(preprocess_2d_path, dataset),
            decode_trfrm(config['preprocess_3d']).from_file(preprocess_3d_path, dataset),
            decode_trfrm(config['preprocess_pcdepth']).from_file(preprocess_pcdepth_path, dataset),
            concat_cdepth, RemoveIndex(), ToTensor()]
