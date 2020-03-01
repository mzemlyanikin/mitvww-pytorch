# Winning solution to Visual Wake Words challenge'19 from MIT in PyTorch

[Model from MIT HAN Lab](https://github.com/mit-han-lab/VWW) implemented in PyTorch.

# Install

Python 3.5+ is required.

`pip install -r requirements.txt`

# Usage

### PyTorch model definition

You can find PyTorch model definition in `mit_vww_pytorch.py`.

### Weights conversion

To convert the model weights you need:

1. Download a saved model `model_fp32.pb` from [author's repository](https://github.com/mit-han-lab/VWW)
2. Run `python load_weights_from_pb.py -m <path to the saved model>`. By default, checkpoint is saved to `mitvww_pytorch.pth`.

Script will return tensors that were not initialized (only `num_batches_tracked` tensors for `BatchNorm` layers in our case. Other tensors should be initialized with TF counterparts)

Script for weights conversion from TF saved model to PyTorch checkpoint should work for different models.
To convert the model you want you need:

1. Implement model in PyTorch with the same structure as TF model.
2. Adjust TF tensors' names to match them with tensors in PyTorch model if needed.

### Check the equality of PyTorch and TF outputs

Optionally, you can run tests with `PYTHONPATH=. pytest` to verify that outputs of PyTorch and TF models are the same.

### Graph visualization

You can save TF graph to visualize it with TensorBoard: 
`python visualize.py -m <path to the saved model> -s <directory to save tensorboard logs>`.
To visualize graph run: `tensorboard --logdir <directory to save tensorboard logs>`
