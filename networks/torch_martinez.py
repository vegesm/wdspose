import torch.nn as nn

from util.params import Params


class DenseBlock(nn.Module):
    def __init__(self, dense_size, n_layers, activation, dropout, residual_enabled,
                 batchnorm_enabled, layernorm_enabled, normclip_enabled, bn_momentum, bn_affine, name=None):
        super(DenseBlock, self).__init__()
        assert residual_enabled, "residual_enabled==False is not implemented"
        self.residual = DenseBlock._residual_branch(dense_size, n_layers, activation, dropout, residual_enabled,
                                                    batchnorm_enabled, layernorm_enabled, normclip_enabled, bn_momentum, bn_affine, name)

    @staticmethod
    def _residual_branch(dense_size, n_layers, activation, dropout, residual_enabled,
                         batchnorm_enabled, layernorm_enabled, normclip_enabled, bn_momentum, bn_affine, name=None):

        assert not normclip_enabled

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(dense_size, dense_size))
            if batchnorm_enabled:
                layers.append(
                    nn.BatchNorm1d(dense_size, momentum=bn_momentum, affine=bn_affine))  # Note track_running_stats should be False in eval?
            if layernorm_enabled:
                layers.append(nn.LayerNorm(dense_size))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                raise Exception('Unimplemented activation: ' + activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.residual(x)
        out += x

        return out


def martinez_net(params, input_size, output_size):
    layers = []
    layers.append(nn.Linear(input_size, params.dense_size))
    # Originally this was left out, for backward compatibility there is a setting for it
    if params.first_dense_full:
        if params.batchnorm_enabled:
            layers.append(nn.BatchNorm1d(params.dense_size, momentum=params.bn_momentum,
                                         affine=params.bn_affine))  # Note track_running_stats should be False in eval?
        if params.layernorm_enabled:
            layers.append(nn.LayerNorm(params.dense_size))

        # layers.append(nn.ReLU())
        #
        if params.dropout > 0 and params.use_first_dropout:
            layers.append(nn.Dropout(params.dropout))

    for _ in range(params.n_blocks_in_model):
        layers.append(DenseBlock(params.dense_size, params.n_layers_in_block, params.activation,
                                 params.dropout, params.residual_enabled, params.batchnorm_enabled,
                                 params.layernorm_enabled, params.normclip_enabled, params.bn_momentum, params.bn_affine))
    layers.append(nn.Linear(params.dense_size, output_size))

    model = nn.Sequential(*layers)

    # initialize weights
    for m in model.modules():
        #         print m
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return "martinez", model


def default_config():
    p = Params()

    p.normclip_enabled = False
    p.dense_size = 1024
    p.n_layers_in_block = 2

    p.activation = 'relu'
    p.dropout = 0.5
    p.residual_enabled = True
    p.batchnorm_enabled = True
    p.n_blocks_in_model = 2

    p.layernorm_enabled = False
    p.first_dense_full = False  # True is better, this is for backward compatibility

    return p
