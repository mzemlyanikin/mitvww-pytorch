import argparse

import tensorflow as tf
from tensorflow.python.platform import gfile


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize graph of the saved model")
    parser.add_argument('-m', '--model-filename', type=str, help='Path to the saved model',
                        default="model_fp32.pb")
    parser.add_argument('-s', '--save-dir', type=str, help='Directory to write graph', default='logs')
    return parser.parse_args()


def visualize_graph(model_filename, logs_path):
    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)

    train_writer = tf.summary.FileWriter(logs_path)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()


def main():
    args = parse_args()
    visualize_graph(args.model_filename, args.save_dir)


if __name__ == '__main__':
    main()