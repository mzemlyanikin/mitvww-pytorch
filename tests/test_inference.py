import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.platform import gfile
import torch

from mit_vww_pytorch import MITVWW


@pytest.fixture
def pt_model():
    model = MITVWW()
    model.load_state_dict(torch.load('pytorch_mitvww.pth'))
    return model


def test_pytorch_vs_tf_inference(pt_model, saved_model='model_fp32.pb',
                                 input_shape=None, input_node_name='import/input:0',
                                 output_node_name='import/Softmax:0'):
    if input_shape is None:
        input_shape = [1, 3, 208, 238]
    else:
        if not isinstance(input_shape, (list, tuple)):
            raise TypeError('Input shape should be tuple or list')

    dummy_input = torch.randn(*input_shape)
    pt_model.eval()
    np_torch_out = pt_model(dummy_input).data.numpy()

    with tf.Session() as sess:
        with gfile.FastGFile(saved_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)

        x_tensor = sess.graph.get_tensor_by_name(input_node_name)
        output = sess.graph.get_tensor_by_name(output_node_name)
        feed_dict = {x_tensor: np.transpose(dummy_input.data.numpy(), [0, 2, 3, 1])}
        tf_output = sess.run([output], feed_dict=feed_dict)
    if len(np_torch_out.shape) == 4:
        np_torch_out = np.transpose(np_torch_out.data.numpy(), [0, 2, 3, 1])

    assert np_torch_out.shape == tf_output[0].shape
    np.testing.assert_almost_equal(np_torch_out, tf_output[0], decimal=5)
