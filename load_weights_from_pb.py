import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import torch

from mit_vww_pytorch import MITVWW


def parse_args():
    parser = argparse.ArgumentParser(description="Create PyTorch checkpoint from TF saved model")
    parser.add_argument('-m', '--model-filename', type=str, help='Path to the saved model',
                        default='model_fp32.pb')
    parser.add_argument('-s', '--save-filename', type=str, help='Path to write PyTorch checkpoint',
                        default='pytorch_mitvww.pth')
    return parser.parse_args()


def correct_node_name(old_name, substrs_to_replace):
    layer_name = old_name
    for (old_value, new_value) in substrs_to_replace:
        layer_name = layer_name.replace(old_value, new_value)
    return layer_name


def tf_weights_to_pytorch(tf_tensor):
    value = tensor_util.MakeNdarray(tf_tensor)
    if len(value.shape) == 4:  # transpose conv weights
        # TF conv weight format [filter_height, filter_width, in_channels, out_channels]
        # PyTorch conv weight format [out_channels, in_channels, kernel_height, kernel_width]
        value = np.transpose(value, [3, 2, 0,
                                     1])
        if value.shape[0] == 1:
            # Depthwise conv weights in TF [filter_height, filter_width, in_channels, channel_multiplier=1]
            # PyTorch depwthvise conv weights format [out_channels, in_channels/groups=1, filter_height, filter_width
            value = np.transpose(value, [1, 0, 2,
                                         3])
    elif len(value.shape) == 2:  # weights conversion for FC layer
        value = np.transpose(value)
    return torch.Tensor(value.astype(float))


def initialize_pt_model_with_tf_pb(model, pb_filename, layers_to_rename):
    with tf.Session() as sess:
        with gfile.FastGFile(pb_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
            graph_nodes = [n for n in graph_def.node]
            wts = [n for n in graph_nodes if n.op == 'Const']

    model_dict = model.state_dict()

    pretrained_dict = {}
    for node in wts:
        layer_name = correct_node_name(node.name, layers_to_rename)
        if layer_name in model_dict:
            pretrained_dict[layer_name] = tf_weights_to_pytorch(node.attr['value'].tensor)

    for k, v in pretrained_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    not_initialized = set()
    for k, v in model_dict.items():
        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
            not_initialized.add(k)
    print('Not initialized tensors:', sorted(not_initialized))

    model.load_state_dict(model_dict)


if __name__ == '__main__':
    args = parse_args()
    substrs_to_replace = [('mobile_inverted_conv/', ''),
                        ('BatchNorm/', ''),
                        ('classifier/', ''),
                        ('moving_mean', 'running_mean'),
                        ('moving_variance', 'running_var'),
                        ('bn/beta', 'bn/bias'),
                        ('bn/gamma', 'bn/weight'),
                        ('/', '.')]

    model = MITVWW()
    initialize_pt_model_with_tf_pb(model, args.model_filename, substrs_to_replace)
    torch.save(model.state_dict(), args.save_filename)
